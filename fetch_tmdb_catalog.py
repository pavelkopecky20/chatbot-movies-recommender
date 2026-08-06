"""
Stáhne rozšířený katalog filmů z TMDB (The Movie Database) API a namapuje ho
na stejnou CatalogItem strukturu, jakou používá brain.py (id, title, genre,
origin_country, year, description, keywords, cast, providers).

Vyžaduje TMDB_API_KEY v .env -- zdarma na https://www.themoviedb.org/settings/api
(zaregistrovat účet -> Settings -> API -> Request API key, typ "Developer").

Výstup: catalog_tmdb.json ve stejné složce; brain.py ho automaticky načítá.

Pro každý film se volá /discover/movie (objevení ID, pár desítek volání na stránky)
a pak /movie/{id} s append_to_response=keywords,credits,watch/providers
(jedno volání NA FILM navíc) -- při výchozích počtech stránek (cíl ~1000-2000
titulů ve výsledném katalogu) trvá celý běh řádově 20-25 minut.
"""

import json
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from brain import CatalogItem  # znovupoužije stejnou datovou strukturu jako hlavní prototyp

load_dotenv()

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")            # klíč se čte z .env, stejně jako OPENAI_API_KEY v brain.py
TMDB_BASE_URL = "https://api.themoviedb.org/3"
WATCH_REGION = "CZ"                                        # streamovací dostupnost nás zajímá pro český trh


def _get(path: str, params: dict, language: str = "cs-CZ") -> dict:
    """Jedno GET volání na TMDB API s API klíčem; language jde přepsat (viz _load_genre_map)."""
    query = {**params, "api_key": TMDB_API_KEY, "language": language}
    response = requests.get(f"{TMDB_BASE_URL}{path}", params=query, timeout=10)
    response.raise_for_status()                            # při chybě (401, 429, ...) rovnou spadne s jasnou hláškou
    return response.json()


def _load_genre_map() -> dict[int, str]:
    """
    TMDB vrací žánry jako id -- tohle je jejich převod na jména.
    Natvrdo v en-US, i když zbytek dat je v češtině -- brain.py a embedding_classifier.py
    počítají s anglickými nálepkami žánrů (genre="comedy", ne "komedie"), musí sedět.
    """
    data = _get("/genre/movie/list", {}, language="en-US")
    return {g["id"]: g["name"].lower() for g in data["genres"]}


def _find_spy_keyword_id() -> Optional[int]:
    """TMDB nemá nativní žánr 'spy' (na rozdíl od katalogu v brain.py) -- hledá se přes keyword."""
    data = _get("/search/keyword", {"query": "spy"})
    results = data.get("results", [])
    return results[0]["id"] if results else None


def _fetch_pages(path: str, params: dict, pages: int) -> list[dict]:
    """TMDB stránkuje po 20 výsledcích -- stáhne zadaný počet stránek a spojí je."""
    movies = []
    for page in range(1, pages + 1):
        data = _get(path, {**params, "page": page})
        movies.extend(data.get("results", []))
        time.sleep(0.25)                                    # šetrné tempo vůči rate limitu API
    return movies


def _fetch_movie_detail(movie_id: int) -> dict:
    """Plný detail filmu včetně keywords/credits/watch-providers v jednom volání (append_to_response)."""
    return _get(f"/movie/{movie_id}", {"append_to_response": "keywords,credits,watch/providers"})


def _extract_origin_country(detail: dict, fallback: str) -> str:
    """Skutečná země původu z detailu -- fallback je jen odhad z toho, kterým discover dotazem byl film nalezen."""
    codes = detail.get("origin_country") or []
    if codes:
        return codes[0]
    countries = detail.get("production_countries") or []
    if countries:
        return countries[0]["iso_3166_1"]
    return fallback


def _extract_keywords(detail: dict) -> list[str]:
    return [k["name"] for k in detail.get("keywords", {}).get("keywords", [])]


def _extract_cast(detail: dict, limit: int = 5) -> list[str]:
    """Jen top N podle 'order' (pořadí v titulkách) -- ne celý cast, ten má klidně 50+ jmen."""
    cast = detail.get("credits", {}).get("cast", [])
    top_cast = sorted(cast, key=lambda c: c.get("order", 999))[:limit]
    return [c["name"] for c in top_cast]


def _extract_providers(detail: dict, region: str = WATCH_REGION) -> list[str]:
    """Streamovací platformy pro daný region -- flatrate (předplatné) + rent + buy, deduplikované."""
    region_data = detail.get("watch/providers", {}).get("results", {}).get(region, {})
    names = [
        p["provider_name"]
        for key in ("flatrate", "rent", "buy")
        for p in region_data.get(key, [])
    ]
    seen = set()
    deduped = []
    for name in names:                                       # ruční dedup -- zachová pořadí, set() by ho promíchal
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def _build_catalog_item(
    movie_id: int,
    raw: dict,
    detail: dict,
    genre_map: dict[int, str],
    origin_hint: str,
    forced_genre: Optional[str],
) -> Optional[CatalogItem]:
    """Spojí hrubá data z discover (title/overview/release_date) s detailem (keywords/cast/providers/origin)."""
    title = raw.get("title")
    overview = raw.get("overview")
    release_date = raw.get("release_date")
    if not title or not overview or not release_date:      # radši méně titulů než prázdné popisy
        return None

    genre_ids = [g["id"] for g in detail.get("genres", [])]
    genre = forced_genre or next(                            # forced_genre = "spy" u výsledků z keyword hledání
        (genre_map[gid] for gid in genre_ids if gid in genre_map), "drama"
    )

    return CatalogItem(
        id=f"tmdb-{movie_id}",                                # prefix "tmdb-" -- nekoliduje s ručně psanými t001..t006
        title=title,
        genre=genre,
        origin_country=_extract_origin_country(detail, fallback=origin_hint),
        year=int(release_date[:4]),
        description=overview,
        keywords=_extract_keywords(detail),
        cast=_extract_cast(detail),
        providers=_extract_providers(detail),
    )


def fetch_catalog(pages_local: int = 25, pages_international: int = 45, pages_spy: int = 17) -> list[CatalogItem]:
    # 25+45+17 stránek * 20 = ~1740 surových záznamů před dedupem/filtrem -- cílí na finální katalog ~1000-2000 titulů.
    # Detail fáze (jedno volání NA FILM) je při tomhle objemu dlouhá -- řádově 20 minut, viz komentář v docstringu nahoře.
    if not TMDB_API_KEY:
        raise RuntimeError(
            "Chybí TMDB_API_KEY v .env. Zdarma ho získáš na "
            "https://www.themoviedb.org/settings/api."
        )

    genre_map = _load_genre_map()
    spy_keyword_id = _find_spy_keyword_id()

    # Fáze 1 -- discover: jen objevení ID + hrubá data, ať víme, KTERÉ filmy vůbec chceme.
    stubs: dict[int, dict] = {}                              # movie_id -> {"raw", "origin_hint", "forced_genre"}

    local_raw = _fetch_pages(                                 # production_countries obsahuje CZ -- ověřený origin hint
        "/discover/movie",
        {"with_origin_country": "CZ", "sort_by": "popularity.desc"},
        pages_local,
    )
    for m in local_raw:
        stubs.setdefault(m["id"], {"raw": m, "origin_hint": "CZ", "forced_genre": None})

    intl_raw = _fetch_pages(                                   # bez country filtru -- origin_hint je jen odhad, detail ho upřesní
        "/discover/movie",
        {"sort_by": "popularity.desc"},
        pages_international,
    )
    for m in intl_raw:
        stubs.setdefault(m["id"], {"raw": m, "origin_hint": "XX", "forced_genre": None})

    if spy_keyword_id:                                         # spy žánr dohledaný přes keyword, ne přes genre_ids
        spy_raw = _fetch_pages(
            "/discover/movie",
            {"with_keywords": spy_keyword_id, "sort_by": "popularity.desc"},
            pages_spy,
        )
        for m in spy_raw:
            entry = stubs.setdefault(m["id"], {"raw": m, "origin_hint": "XX", "forced_genre": None})
            entry["forced_genre"] = "spy"                       # i když se film našel i jinde, spy klasifikace má přednost

    print(f"[FETCH] Nalezeno {len(stubs)} unikátních filmů, stahuji detail (keywords/cast/providers) pro každý...")

    # Fáze 2 -- detail: jedno volání NA FILM navíc, odtud jde vzít keywords/cast/providers/origin_country.
    catalog: list[CatalogItem] = []
    for i, (movie_id, stub) in enumerate(stubs.items(), start=1):
        try:
            detail = _fetch_movie_detail(movie_id)
        except requests.RequestException as exc:                # jeden neúspěšný film nesmí shodit celý běh
            print(f"[WARNING] Detail pro film {movie_id} selhal ({exc}), přeskočeno.")
            continue
        time.sleep(0.25)                                         # stejné šetrné tempo jako u discover stránek

        item = _build_catalog_item(
            movie_id, stub["raw"], detail, genre_map, stub["origin_hint"], stub["forced_genre"]
        )
        if item:
            catalog.append(item)

        if i % 20 == 0:                                           # průběžný log -- běh trvá desítky sekund až pár minut
            print(f"[FETCH] ... {i}/{len(stubs)} zpracováno")

    return catalog


def save_catalog(catalog: list[CatalogItem], path: str = "catalog_tmdb.json") -> None:
    payload = [item.__dict__ for item in catalog]               # CatalogItem je dataclass -- __dict__ dá čistý JSON-serializovatelný dict
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)     # ensure_ascii=False -- ať se čeština uloží čitelně, ne jako \uXXXX
    print(f"Uloženo {len(payload)} titulů do {path}")


if __name__ == "__main__":
    catalog = fetch_catalog()
    save_catalog(catalog)
