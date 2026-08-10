"""
Streamlit frontend nad brain.py -- veřejná portfolio ukázka na Hugging Face Spaces.
Veškerá doménová logika (retrieval, sticky constraints, LLM routing) zůstává
v brain.py beze změny; tenhle soubor jen řeší UI, per-session konverzační stav
a kvótu zpráv (ta je záměrně sdílená napříč sessions, viz komentář u _get_quota_store).
"""

import os
import threading
import uuid

import streamlit as st
from openai import OpenAI
from streamlit_cookies_controller import CookieController

import brain
from embedding_classifier import ConstraintClassifier

st.set_page_config(page_title="Movie Chatbot", page_icon="🎬")

FREE_QUOTA = 3      # zpráv zdarma na návštěvníka -- viz OpenAI spend cap jako druhá vrstva ochrany
EXTENDED_QUOTA = 50  # limit pro návštěvníky s platným přístupovým kódem (viz _try_unlock)

# Kvóta je záměrně MIMO st.session_state -- ten se resetuje při obyčejném refreshi stránky
# (nová WebSocket session = nová session_state), takže by to byl triviální obchvat. Místo
# toho je to sdílený stav na úrovni procesu, klíčovaný podle _client_key() (viz níže --
# cookie v prohlížeči, ne IP adresa/hlavičky, ty se na Streamlit Community Cloud ukázaly
# jako nespolehlivé). Přežije refresh, ale ne restart/redeploy Space (přijatelný kompromis
# pro demo, backstop je OpenAI spend cap).
#
# DŮLEŽITÉ: obyčejný modulový `dict = {}` by NEFUNGOVAL -- Streamlit přeexekuuje celý
# top-level kód skriptu při KAŽDÉM rerunu (i implicitním po st.rerun()), takže by se
# vynuloval při každé zprávě. @st.cache_resource je jediný způsob, jak mít objekt,
# co skutečně přežije reruny i je sdílený napříč sessions.
_quota_lock = threading.Lock()


@st.cache_resource
def _get_quota_store() -> dict[str, int]:
    return {}


_VISITOR_COOKIE_NAME = "chatbot_visitor_id"

# IP/X-Forwarded-For se v praxi ukázaly jako nespolehlivé (na Streamlit Community Cloud
# dávaly nestabilní hodnoty napříč refreshi stránky téhož návštěvníka). Ruční JS injekce
# přes st.iframe/document.cookie TAKY nefungovala spolehlivě -- moderní prohlížeče čím
# dál víc partitionují cookies/storage nastavené UVNITŘ iframe od cookies hlavní stránky,
# takže se cookie nastavená z iframu k serveru na dalším requestu vůbec nedostala (ověřeno
# v DevTools -- při každém refreshi vznikalo nové ID). CookieController je skutečná
# obousměrná Streamlit komponenta (vlastní frontend build, ne vystřelený <script> bez
# zpětné vazby), takže cookie nastavuje/čte spolehlivě na správné doméně.
_cookie_controller = CookieController()


def _get_or_create_visitor_id() -> str:
    """
    RACE CONDITION, na kterou je potřeba dávat pozor: CookieController je async
    komponenta -- na úplně první vykreslení nové session (= po refreshi stránky)
    její getAll() vždy vrátí jen prázdný `default={}`, i když prohlížeč reálně
    nějakou cookie z minula pošle. Skutečná hodnota z JS dorazí a projeví se
    až o jeden rerun později. Kdybychom na ten prázdný výsledek reagovali hned
    (vygenerovat nové ID a přepsat cookie), přepsali bychom platnou starou cookie
    dřív, než by vůbec měla šanci se načíst -- to je přesně to, proč se ID měnilo
    při KAŽDÉM refreshi. Proto se čeká jeden rerun navíc (_cookie_settle_done),
    než se appka rozhodne, že jde o nového návštěvníka.

    Nově vygenerované ID se mezitím drží v st.session_state, ať je stabilní
    v rámci aktuální session. Když má prohlížeč cookies vypnuté, degraduje se to
    zpět na chování per-session (kvóta se resetuje při refreshi) -- nespadne to,
    jen ztratí výhodu.

    Volání CookieController jsou obalená try/except -- v produkci se ukázalo,
    že interní stav komponenty (self.__cookies) může za určitých okolností být
    None místo slovníku (bug/edge-case v knihovně samotné, ne v našem kódu),
    což shazovalo celou appku na TypeError. Radši degradovat na per-session
    chování než appku úplně zabít kvůli závislosti, do jejíhož vnitřku nevidíme.
    """
    try:
        cookie_id = _cookie_controller.get(_VISITOR_COOKIE_NAME)
    except Exception as exc:
        print(f"[COOKIE] CookieController.get() selhalo ({exc}) -- pokračuju bez cookie.")
        cookie_id = None

    if cookie_id:
        return cookie_id

    if "temp_visitor_id" in st.session_state:            # v týhle session už jsme se jednou rozhodli, drž se toho
        return st.session_state.temp_visitor_id

    if not st.session_state.get("_cookie_settle_done"):    # dej komponentě šanci vrátit REÁLNOU hodnotu, ne jen default
        st.session_state._cookie_settle_done = True
        st.rerun()

    st.session_state.temp_visitor_id = str(uuid.uuid4())
    try:
        _cookie_controller.set(
            _VISITOR_COOKIE_NAME,
            st.session_state.temp_visitor_id,
            max_age=31536000,  # 1 rok, v sekundách
        )
    except Exception as exc:
        print(f"[COOKIE] CookieController.set() selhalo ({exc}) -- ID bude platit jen pro tuhle session.")
    return st.session_state.temp_visitor_id


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

st.title("🎬 Movie Chatbot")
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
            with st.container(border=True):
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
