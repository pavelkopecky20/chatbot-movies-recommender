"""
Kompletní implementace embedding klasifikace pro extrakci sticky constraints.
Náhrada/doplněk za čistý keyword matching z hlavního prototypu.

Funguje ve dvou režimech:
- Se skutečným OPENAI_API_KEY: použije text-embedding-3-small (multilingual)
- Bez klíče: fallback (jen pro spustitelnost, NENÍ sémanticky funkční -- viz demo níže)
"""

import os
import numpy as np
from typing import Optional

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False


# ---------------------------------------------------------------------------
# 1. Embedding provider (stejný princip jako v hlavním prototypu)
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    def __init__(self, client: Optional["OpenAI"]):
        self.client = client

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.client is not None:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return np.array([d.embedding for d in response.data])

        # Fallback -- jen pro spustitelnost bez API klíče, NENÍ sémanticky funkční
        vectors = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vectors.append(rng.random(64))
        return np.array(vectors)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# 2. Definice referenčních frází pro každou dimenzi constraintu
#    -- tohle je ta "few-shot" sada, kterou bys ladil na reálných datech
# ---------------------------------------------------------------------------

ORIGIN_REFERENCE = {
    "local": [
        "chci lokální obsah",
        "jen české produkce",
        "domácí tvorba",
        "tuzemský film",
        "chci jen z Česka",
        "něco od nás",
        "nechci zahraniční",
    ],
    "international": [
        "zahraniční filmy",
        "cizí produkce",
        "mezinárodní tituly",
        "chci něco ze zahraničí",
        "něco ze světa",
        "nechci český",
    ],
}

GENRE_REFERENCE = {
    "spy": [
        "špionážní film",
        "agent a tajné služby",
        "studenoválečný thriller",
        "film o špiónech",
    ],
    "comedy": [
        "veselá komedie",
        "chci se zasmát",
        "něco zábavného a lehkého",
        "chci film, u kterého se zasměju",
        "chci se u filmu bavit",
        "chci něco vtipného",
    ],
    "drama": [
        "vážné drama",
        "emotivní příběh",
        "rodinné drama",
    ],
    "horror": [
        "strašlivý film",
        "horrorní příběh",
        "něco děsivého",
        "chci se bát",
        "chci horror",
    ]
}

SORT_REFERENCE = {
    "year_desc": [
        "nejnovější film",
        "chci něco aktuálního",
        "poslední dobou vyšlé tituly",
    ],
}

CAST_REFERENCE = {}

# ---------------------------------------------------------------------------
# 3. Klasifikátor -- centroidy se počítají JEDNOU při inicializaci (ne za běhu)
# ---------------------------------------------------------------------------

class ConstraintClassifier:
    """
    Pro každou dimenzi (origin, genre, sort_by) drží centroid každé kategorie.
    Centroid = průměrný embedding všech referenčních frází dané kategorie.
    """

    def __init__(self, embedder: EmbeddingProvider, threshold: float = 0.6):
        self.embedder = embedder
        self.threshold = threshold

        # Předpočítá centroidy pro každou dimenzi -- děje se jen JEDNOU při startu
        self.origin_centroids = self._build_centroids(ORIGIN_REFERENCE)
        self.genre_centroids = self._build_centroids(GENRE_REFERENCE)
        self.sort_centroids = self._build_centroids(SORT_REFERENCE)

        self.cast_centroids = self._build_centroids(CAST_REFERENCE)  # Přidáno pro detekci herců
        
    def _build_centroids(self, reference: dict[str, list[str]]) -> dict[str, np.ndarray]:
        centroids = {}
        for label, phrases in reference.items():
            vecs = self.embedder.embed(phrases)       # zaembeduje všechny fráze kategorie
            centroids[label] = vecs.mean(axis=0)       # zprůměruje je do jednoho bodu
        return centroids

    def _classify(self, msg_vec: np.ndarray, centroids: dict[str, np.ndarray]) -> tuple[Optional[str], float]:
        best_label, best_score = None, -1.0
        for label, centroid in centroids.items():
            score = cosine_sim(msg_vec, centroid)
            if score > best_score:
                best_label, best_score = label, score
        if best_score < self.threshold:
            return None, best_score               # "nejsem si dost jistý" -- vrátí None
        return best_label, best_score

    def classify_message(self, user_message: str) -> dict:
        """
        Vrátí slovník s výsledky klasifikace napříč VŠEMI dimenzemi najednou.
        Runtime cena: JEDNO embedding volání na zprávu (ne jedno na dimenzi).
        """
        msg_vec = self.embedder.embed([user_message])[0]   # jediné volání za běhu

        origin_label, origin_score = self._classify(msg_vec, self.origin_centroids)
        genre_label, genre_score = self._classify(msg_vec, self.genre_centroids)
        sort_label, sort_score = self._classify(msg_vec, self.sort_centroids)

        result = {}
        if origin_label:
            result["origin"] = origin_label
        if genre_label:
            result["genre"] = genre_label
        if sort_label:
            result["sort_by"] = sort_label

        return result, {
            "origin_score": origin_score,
            "genre_score": genre_score,
            "sort_score": sort_score,
            
        }


# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if (_openai_available and api_key) else None

    if client is None:
        print("=== FALLBACK režim (bez OPENAI_API_KEY) ===")
        print("POZOR: výsledky níže NEBUDOU sémanticky správné -- fallback je náhodný.")
        print("Se skutečným API klíčem by klasifikace fungovala smysluplně.\n")

    embedder = EmbeddingProvider(client)
    classifier = ConstraintClassifier(embedder, threshold=0.75)

    test_messages = [
        "chci jen tuzemskou tvorbu",              # mělo by dát origin=local
        "něco jako ten starý film s agentem",      # mělo by dát genre=spy
        "chci něco aktuálního, co teď vyšlo",       # mělo by dát sort_by=year_desc
        "nechci nic ze zahraničí",                  # mělo by dát origin=local (přes zápor)
    ]

    for msg in test_messages:
        constraints, scores = classifier.classify_message(msg)
        print(f"Zpráva: '{msg}'")
        print(f"  Constrainty: {constraints}")
        print(f"  Skóre:       {scores}")
        print()
