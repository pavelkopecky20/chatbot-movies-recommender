"""
Streamlit frontend nad brain.py -- veřejná portfolio ukázka na Hugging Face Spaces.
Veškerá doménová logika (retrieval, sticky constraints, LLM routing) zůstává
v brain.py beze změny; tenhle soubor jen řeší UI, per-session konverzační stav
a kvótu zpráv (ta je záměrně sdílená napříč sessions, viz komentář u _quota_by_ip).
"""

import os
import threading

import streamlit as st
from openai import OpenAI

import brain
from embedding_classifier import ConstraintClassifier

st.set_page_config(page_title="Movie Chatbot", page_icon="🎬")

FREE_QUOTA = 3  # zpráv zdarma na návštěvníka -- viz OpenAI spend cap jako druhá vrstva ochrany

# Kvóta je záměrně MIMO st.session_state -- ten se resetuje při obyčejném refreshi stránky
# (nová WebSocket session = nová session_state), takže by to byl triviální obchvat. Místo
# toho je to sdílený stav na úrovni procesu, klíčovaný podle IP adresy -- přežije refresh,
# ale ne restart/redeploy Space (přijatelný kompromis pro demo, backstop je OpenAI spend cap).
# Sdílená IP (NAT, VPN, firemní síť) = sdílená kvóta -- taky přijatelný kompromis pro demo.
#
# DŮLEŽITÉ: obyčejný modulový `dict = {}` by NEFUNGOVAL -- Streamlit přeexekuuje celý
# top-level kód skriptu při KAŽDÉM rerunu (i implicitním po st.rerun()), takže by se
# vynuloval při každé zprávě. @st.cache_resource je jediný způsob, jak mít objekt,
# co skutečně přežije reruny i je sdílený napříč sessions.
_quota_lock = threading.Lock()


@st.cache_resource
def _get_quota_store() -> dict[str, int]:
    return {}


def _client_key() -> str:
    """
    Streamlit `ip_address` vrací None při přístupu přes localhost/loopback --
    přesně tahle situace může nastat na HF Spaces, kde interní proxy k appce
    přistupuje přes localhost, takže `ip_address` by pro všechny návštěvníky
    vracelo None (kvóta by se zdegradovala na jeden sdílený pool pro všechny --
    přísnější, ne rozbité, ale ne zamýšlené chování). X-Forwarded-For hlavičku
    nastavují běžné reverse proxy (včetně HF), takže se zkouší přednostně.
    """
    forwarded_for = st.context.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()  # první IP v řetězci = skutečný klient, zbytek jsou proxy
    return st.context.ip_address or "unknown"


def _messages_used() -> int:
    return _get_quota_store().get(_client_key(), 0)


def _increment_messages_used() -> None:
    key = _client_key()
    with _quota_lock:
        quota_store = _get_quota_store()
        quota_store[key] = quota_store.get(key, 0) + 1


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

    if _messages_used() >= FREE_QUOTA:                          # kvóta vyčerpaná -- ŽÁDNÉ volání OpenAI
        response = brain.AgentResponse(
            reply=(
                f"Vyčerpal jsi bezplatný limit {FREE_QUOTA} zpráv pro tuhle ukázku. "
                "Zkus to prosím později -- limit je natrvalo na tohle připojení, refresh stránky ho neobnoví. "
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


with st.sidebar:
    st.subheader("Aktivní filtry")
    constraints = st.session_state.conv_state.sticky_constraints
    if constraints:
        for key, value in constraints.items():
            st.write(f"**{key}**: {value}")
    else:
        st.caption("Žádné aktivní omezení.")

    st.divider()
    remaining = max(0, FREE_QUOTA - _messages_used())
    st.caption(f"Zbývá {remaining}/{FREE_QUOTA} zpráv zdarma pro tohle připojení.")

    if st.button("Resetovat konverzaci"):
        st.session_state.conv_state.sticky_constraints.clear()  # kvóta žije mimo session_state, tímhle se nedotkne
        st.session_state.conv_state.history.clear()
        st.session_state.display_log = []
        st.rerun()

    if client is None:
        st.warning("Běží ve fallback režimu (chybí OPENAI_API_KEY) -- odpovědi nejsou generované LLM.")

st.title("🎬 Movie Chatbot")
st.caption(
    "Doporučovací chatbot nad katalogem filmů (TMDB, 170 titulů) -- embedding retrieval, "
    "sticky constraints extrahované klasifikátorem nad embeddingy, LLM routing mezi gpt-4o/gpt-4o-mini."
)

if st.session_state.pending_chip:                              # klik na chip = jako by uživatel tohle napsal
    chip_text = st.session_state.pending_chip
    st.session_state.pending_chip = None
    process_message(chip_text)
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
