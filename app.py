"""
Streamlit frontend nad brain.py -- veřejná portfolio ukázka na Hugging Face Spaces.
Veškerá doménová logika (retrieval, sticky constraints, LLM routing) zůstává
v brain.py beze změny; tenhle soubor jen řeší UI, per-session konverzační stav
a kvótu zpráv (ta je záměrně sdílená napříč sessions, viz komentář u _get_quota_store).
"""

import base64
import os
import threading
import uuid
from pathlib import Path

import streamlit as st
from openai import OpenAI

import brain
from embedding_classifier import ConstraintClassifier

st.set_page_config(page_title="Movie Chatbot", page_icon="🎬")


def _set_element_background(selector: str, image_path: Path, mime: str = "image/jpeg") -> None:
    """Nastaví obrázek na pozadí libovolnému elementu podle CSS selektoru (sidebar, hlavní plocha, ...)."""
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        {selector} {{
            background-image: url("data:{mime};base64,{encoded}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: scroll !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_PICTURES_DIR = Path(__file__).parent / "pictures"  # cesta relativní k app.py -- funguje bez ohledu na to, odkud se streamlit spouští
_set_element_background('[data-testid="stSidebar"]', _PICTURES_DIR / "movies.jpg")
_set_element_background('[data-testid="stAppScrollToBottomContainer"]', _PICTURES_DIR / "popcorn_girls.jpg", mime="image/jpeg")  # skutečný celoplošný panel bez paddingu (třída "stMain", ne testid) -- stMainBlockContainer je jen užší odsazený vnitřní box, proto zůstával bílý pás kolem; ověřeno naživo přes Playwright
# stHeader (horní lišta s "Deploy") a stBottom (spodní pruh s chat_input) jsou SAMOSTATNÉ elementy,
# ne potomci stAppScrollToBottomContainer výš -- proto obrázek odtamtud na ně nedosáhl, potřebují ho zvlášť.
_set_element_background('[data-testid="stHeader"]', _PICTURES_DIR / "movies.jpg", mime="image/jpg")  # defaultně plná bílá + vysoký z-index, obrázek (neprůhledný) ji přebije, protože se kreslí nad barvou
_set_element_background('[data-testid="stBottom"]', _PICTURES_DIR / "popcorn.jpg", mime="image/jpg")

st.markdown(
    """
    <style>
    /* Sidebar má obrázek na pozadí -- text v něm potřebuje vlastní čitelnou podložku, ne přímo na obrázku.
       stSidebarUserContent = přesně ten obsah, co appka do sidebaru sama vkládá (ne navigace/logo apod.). */
    [data-testid="stSidebarUserContent"] {
        background-color: rgba(0, 0, 0, 0.88) !important;
        border-radius: 10px;
        padding: 1rem !important;
    }
    [data-testid="stSidebarUserContent"] * {
        color: white !important;
    }
    /* Tlačítka zvlášť -- vestavěné bílé/světlé pozadí tlačítka + bílý text výš by dalo bílou na bílé,
       nečitelné. Průhledné pozadí + bílý rámeček, žádná plná výplň. */
    [data-testid="stSidebarUserContent"] button {
        background-color: transparent !important;
        border: 2px solid white !important;
        color: white !important;
    }
    [data-testid="stSidebarUserContent"] button:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-color: white !important;
    }
    [data-testid="stCaptionContainer"] {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    /* Hlavní okno má obrázek (old.png) VIDITELNÝ na pozadí (nastaveno na stAppScrollToBottomContainer
       výš, ne na tenhle kontejner -- ten je jen užší odsazený vnitřní box, viz komentář nahoře).
       Sem patří jen čitelnost TEXTU: obecně bílý + stín (čitelný i přímo na obrázku) a jen
       JEDNOTLIVÉ prvky (nadpis, chatové bubliny, karty filmů) mají svou vlastní tmavou podložku. */
    [data-testid="stMainBlockContainer"] * {
        color: white !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);
    }
    [data-testid="stMainBlockContainer"] button {
        background-color: rgba(0, 0, 0, 0.55) !important;
        border: 2px solid white !important;
        color: white !important;
        text-shadow: none;
    }
    [data-testid="stMainBlockContainer"] button:hover {
        background-color: rgba(255, 255, 255, 0.15) !important;
        border-color: white !important;
    }
    /* Nadpis + popisek nahoře -- vlastní tmavý panel (viz st.container(key="header_banner") v kódu). */
    .st-key-header_banner {
        background-color: rgba(0, 0, 0, 0.9);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    /* Chatové bubliny -- vlastní výrazná tmavá podložka, tady se čte nejvíc textu. */
    [data-testid="stChatMessage"] {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border-radius: 10px;
    }
    /* Karty doporučených filmů -- st.container(border=True, key=f"card-...") v kódu, cíleno přes částečnou shodu třídy. */
    [class*="st-key-card-"] {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border-radius: 10px;
    }
    /* Chat input -- stChatInput/stBottom jsou samo o sobě průhledné (obrázek pod nimi je vidět),
       ale vnitřní box vstupního pole má vlastní vestavěné světle šedé pozadí (rgb(240,242,246)),
       bez vlastního data-testid -- cíleno strukturálně (první potomek stChatInput), ne přes
       nestabilní generovanou CSS třídu. */
    [data-testid="stChatInput"] > div:first-child {
        background-color: rgba(0, 0, 0, 0.6) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: white !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    /* Plocha KOLEM vstupního pole (spodní bar) -- Streamlit tam interně vykresluje vlastní
       neprůhledný bílý panel (bez data-testid, jen generovaná třída "e15ve43o3"), co leží
       NAD stBottom a zakrývá jeho obrázek. Bez zásahu do samotného vstupního pole. */
    .e15ve43o3 {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FREE_QUOTA = 3      # zpráv zdarma na návštěvníka -- viz OpenAI spend cap jako druhá vrstva ochrany
EXTENDED_QUOTA = 50  # limit pro návštěvníky s platným přístupovým kódem (viz _try_unlock)

# Kvóta je záměrně MIMO st.session_state -- ten se resetuje při obyčejném refreshi stránky
# (nová WebSocket session = nová session_state), takže by to byl triviální obchvat. Místo
# toho je to sdílený stav na úrovni procesu, klíčovaný podle _client_key() (viz níže --
# ID v URL query parametru, ne cookie/IP/hlavičky -- ty se v praxi ukázaly jako
# nespolehlivé, viz komentář u _get_or_create_visitor_id). Přežije refresh, ale ne
# restart/redeploy Space (přijatelný kompromis pro demo, backstop je OpenAI spend cap).
#
# DŮLEŽITÉ: obyčejný modulový `dict = {}` by NEFUNGOVAL -- Streamlit přeexekuuje celý
# top-level kód skriptu při KAŽDÉM rerunu (i implicitním po st.rerun()), takže by se
# vynuloval při každé zprávě. @st.cache_resource je jediný způsob, jak mít objekt,
# co skutečně přežije reruny i je sdílený napříč sessions.
_quota_lock = threading.Lock()


@st.cache_resource
def _get_quota_store() -> dict[str, int]:
    return {}


_VISITOR_ID_PARAM = "vid"

# Historie neúspěšných pokusů (ať je jasné, proč tohle a ne něco "chytřejšího"):
# 1) IP/X-Forwarded-For hlavičky -- na Streamlit Community Cloud nestabilní napříč
#    refreshi téhož návštěvníka.
# 2) Ruční JS injekce přes st.iframe/document.cookie -- moderní prohlížeče partitionují
#    cookies nastavené UVNITŘ iframe od cookies hlavní stránky, takže se k serveru
#    na dalším requestu vůbec nedostaly.
# 3) streamlit-cookies-controller (skutečná async komponenta) -- v produkci padalo na
#    TypeError kvůli internímu stavu knihovny (self.__cookies == None), a i po obalení
#    try/except se ID při refreshi pořád měnilo -- systematicky nespolehlivé na tomhle
#    nasazení, ne jen občasná chyba.
#
# st.query_params je čistě nativní Streamlit mechanismus -- žádná async komponenta,
# žádný round-trip, žádná závislost na tom, jak si prohlížeč/platforma řeší cookies.
# ID se zapíše přímo do URL; refresh (F5) načte STEJNOU URL i s parametrem, takže ho
# appka uvidí hned na začátku běhu skriptu, synchronně a spolehlivě. Cena: ID je vidět
# v adresním řádku (jen náhodné UUID, nic citlivého).
def _get_or_create_visitor_id() -> str:
    vid = st.query_params.get(_VISITOR_ID_PARAM)
    if vid:
        return vid
    new_id = str(uuid.uuid4())
    st.query_params[_VISITOR_ID_PARAM] = new_id
    return new_id


def _client_key() -> str:
    return _get_or_create_visitor_id()


def _messages_used() -> int:
    return _get_quota_store().get(_client_key(), 0)


def _increment_messages_used() -> None:
    key = _client_key()
    with _quota_lock:
        quota_store = _get_quota_store()
        quota_store[key] = quota_store.get(key, 0) + 1


# Stejný princip jako u kvóty -- @st.cache_resource, ne obyčejný modulový set,
# jinak by se odemčení ztratilo při každém rerunu (viz komentář výše u kvóty).
@st.cache_resource
def _get_unlocked_store() -> set[str]:
    return set()


def _valid_access_codes() -> set[str]:
    """Kódy žijí ve Streamlit Secrets (stejné místo jako OPENAI_API_KEY), ne v repu."""
    try:
        return set(st.secrets.get("access_codes", {}).values())
    except Exception:
        return set()  # lokální běh bez secrets.toml -- žádné kódy nejsou platné, ne pád appky


def _is_unlocked() -> bool:
    return _client_key() in _get_unlocked_store()


def _try_unlock(code: str) -> bool:
    if code and code in _valid_access_codes():
        _get_unlocked_store().add(_client_key())
        return True
    return False


def _effective_quota() -> int:
    return EXTENDED_QUOTA if _is_unlocked() else FREE_QUOTA


@st.cache_resource(show_spinner="Připravuji katalog a embeddingy...")
def get_resources():
    """Postaví drahé sdílené zdroje JEDNOU za život procesu (napříč všemi návštěvníky)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if (brain._openai_available and api_key) else None
    embedder = brain.EmbeddingProvider(client)
    classifier = ConstraintClassifier(embedder)
    store = brain.VectorStore(brain.CATALOG, embedder)
    catalog_by_id = {c.id: c for c in brain.CATALOG}
    return client, store, classifier, catalog_by_id


client, store, classifier, catalog_by_id = get_resources()

if "conv_state" not in st.session_state:
    st.session_state.conv_state = brain.ConversationState(classifier=classifier)
if "display_log" not in st.session_state:
    st.session_state.display_log = []  # [{"role", "content", "picks", "chips"}, ...] -- bohatší než conv_state.history
if "pending_chip" not in st.session_state:
    st.session_state.pending_chip = None


def process_message(user_message: str) -> None:
    st.session_state.display_log.append(
        {"role": "user", "content": user_message, "picks": [], "chips": []}
    )

    quota = _effective_quota()
    if _messages_used() >= quota:                                # kvóta vyčerpaná -- ŽÁDNÉ volání OpenAI
        response = brain.AgentResponse(
            reply=(
                f"Vyčerpal jsi limit {quota} zpráv pro tuhle ukázku. "
                "Zkus to prosím později -- limit je natrvalo na tohle připojení. "
                "Případně kontaktuj autora této aplikace."
            ),
            picks=[],
            chips=[],
        )
    else:
        response = brain.handle_turn(user_message, st.session_state.conv_state, store, client)
        _increment_messages_used()

    st.session_state.display_log.append(
        {"role": "assistant", "content": response.reply, "picks": response.picks, "chips": response.chips}
    )


def _handle_chip_command(chip_text: str) -> bool:
    """
    CHIP_RESET_ALL / CHIP_RESET_GENRE jsou PŘÍMÉ PŘÍKAZY -- řeší se rovnou tady, ne
    posláním přes process_message()/handle_turn(). Ten dřívější přístup (poslat text
    chipu jako běžnou zprávu a spolehnout se, že ho keyword-matching v update_constraints
    rozpozná) nefungoval spolehlivě -- "Zrušit omezení" (infinitiv) nikdy nesedělo na
    kontrolu "zruš omezení" (rozkazovací způsob) v brain.py. Přímý příkaz navíc nevolá
    OpenAI a nepočítá se do kvóty -- je to čistě UI akce, ne konverzační tah.
    Vrátí True, pokud chip_text rozpoznala jako příkaz (a obsloužila ho), jinak False --
    volající pak pošle text normální cestou přes process_message().
    """
    if chip_text == brain.CHIP_RESET_ALL:
        st.session_state.conv_state.sticky_constraints.clear()
        reply = "Zrušil jsem všechna aktivní omezení."
    elif chip_text == brain.CHIP_RESET_GENRE:
        st.session_state.conv_state.sticky_constraints.pop("genre", None)
        reply = "Zrušil jsem omezení na žánr, ostatní filtry zůstávají beze změny."
    else:
        return False

    st.session_state.display_log.append({"role": "user", "content": chip_text, "picks": [], "chips": []})
    st.session_state.display_log.append({"role": "assistant", "content": reply, "picks": [], "chips": []})
    return True


with st.sidebar:
    st.subheader("Aktivní filtry")
    constraints = st.session_state.conv_state.sticky_constraints
    if constraints:
        for key, value in constraints.items():
            st.write(f"**{key}**: {value}")
    else:
        st.caption("Žádné aktivní omezení.")

    # Trvalá tlačítka, ne jen chips vázané na konkrétní odpověď -- ta se objeví jen
    # u FALLBACK_RESPONSE / "nic jsem nenašel" větve v call_llm(), takže by při normální
    # (úspěšné) odpovědi zmizela. Volají STEJNOU _handle_chip_command() funkci jako chips.
    col_genre, col_all = st.columns(2)
    with col_genre:
        if st.button(brain.CHIP_RESET_GENRE, use_container_width=True):
            _handle_chip_command(brain.CHIP_RESET_GENRE)
            st.rerun()
    with col_all:
        if st.button(brain.CHIP_RESET_ALL, use_container_width=True):
            _handle_chip_command(brain.CHIP_RESET_ALL)
            st.rerun()

    st.divider()
    if _is_unlocked():
        st.caption("✓ Rozšířený přístup aktivní.")
    else:
        entered_code = st.text_input("Přístupový kód (volitelné)", type="password", key="access_code_input")
        if entered_code:
            if _try_unlock(entered_code):
                st.success(f"Kód přijat -- limit navýšen na {EXTENDED_QUOTA} zpráv.")
            else:
                st.error("Neplatný kód.")

    quota = _effective_quota()
    remaining = max(0, quota - _messages_used())
    st.caption(f"Zbývá {remaining}/{quota} zpráv pro tohle připojení.")

    if st.button("Resetovat konverzaci"):
        st.session_state.conv_state.sticky_constraints.clear()  # kvóta žije mimo session_state, tímhle se nedotkne
        st.session_state.conv_state.history.clear()
        st.session_state.display_log = []
        st.rerun()

    if client is None:
        st.warning("Běží ve fallback režimu (chybí OPENAI_API_KEY) -- odpovědi nejsou generované LLM.")

with st.container(key="header_banner"):  # poloprůhledný tmavý panel za nadpisem -- barva/stín textu samotná
    st.title("🎬 Movie Chatbot")           # proti nepředvídatelnému obrázku na pozadí nestačila, tohle je spolehlivé bez ohledu na to, co je pod tím
    st.caption(
        f"Doporučovací chatbot nad katalogem filmů (TMDB, {len(brain.CATALOG)} titulů) -- embedding retrieval, "
        "sticky constraints extrahované klasifikátorem nad embeddingy, LLM routing mezi gpt-4o/gpt-4o-mini."
    )

if st.session_state.pending_chip:                              # klik na chip
    chip_text = st.session_state.pending_chip
    st.session_state.pending_chip = None
    if not _handle_chip_command(chip_text):                    # nejdřív zkusit jako přímý příkaz (reset/žánr)
        process_message(chip_text)                              # jinak poslat jako běžnou zprávu, jako by ji uživatel napsal
    st.rerun()

for i, turn in enumerate(st.session_state.display_log):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

        for pick_id in turn["picks"]:                           # karty doporučených titulů
            item = catalog_by_id.get(pick_id)
            if item is None:
                continue
            with st.container(border=True, key=f"card-{i}-{pick_id}"):  # key -> cílitelná CSS třída st-key-card-... pro čitelnou podložku
                st.markdown(f"**{item.title}** ({item.year}) -- _{item.genre}_")
                description = item.description
                st.caption(description[:200] + ("..." if len(description) > 200 else ""))

        if turn["chips"]:                                       # rychlé odpovědi jako tlačítka
            cols = st.columns(len(turn["chips"]))
            for col, chip in zip(cols, turn["chips"]):
                with col:
                    if st.button(chip, key=f"chip-{i}-{chip}"):
                        st.session_state.pending_chip = chip
                        st.rerun()

prompt = st.chat_input("Napiš, co bys chtěl/a sledovat...")
if prompt:
    process_message(prompt)
    st.rerun()
