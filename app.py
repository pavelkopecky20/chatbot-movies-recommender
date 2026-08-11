"""
Streamlit frontend on top of brain.py -- public portfolio demo on Streamlit Community Cloud.
All domain logic (retrieval, sticky constraints, LLM routing) stays in brain.py unchanged;
this file only handles the UI, per-session conversation state, and the message quota
(intentionally shared across sessions, see the comment on _get_quota_store).
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

st.set_page_config(page_title="Movie Chatbot", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")


def _set_element_background(selector: str, image_path: Path, mime: str = "image/jpeg") -> None:
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


_PICTURES_DIR = Path(__file__).parent / "pictures"
_set_element_background('[data-testid="stSidebar"]', _PICTURES_DIR / "movies.jpg")
st.markdown(
    '<style>[data-testid="stAppScrollToBottomContainer"] { background-color: #000000 !important; }</style>',
    unsafe_allow_html=True,
)
# stHeader and stBottom are separate elements, not descendants of stAppScrollToBottomContainer
# above -- a background set on one doesn't reach them, they need their own.
_set_element_background('[data-testid="stHeader"]', _PICTURES_DIR / "movies.jpg", mime="image/jpg")  # opaque image overrides the default solid-white header because it paints on top of the color layer
_set_element_background('[data-testid="stBottom"]', _PICTURES_DIR / "popcorn.jpg", mime="image/jpg")

st.markdown(
    """
    <style>
    [data-testid="stSidebarUserContent"] {
        background-color: rgba(0, 0, 0, 0.88) !important;
        border-radius: 10px;
        padding: 1rem !important;
    }
    [data-testid="stSidebarUserContent"] * {
        color: white !important;
    }
    /* transparent + white border, not a solid fill -- the built-in button background is light,
       which combined with white text above would be white-on-white */
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
    /* Streamlit renders this container as a flex item with flex-basis: 0% / flex-grow: 1, so
       resizing it directly (width, flex-basis) to break out of stMainBlockContainer's own
       padding (80px sides / 96px top at desktop width) kept getting re-collapsed by that flex
       sizing -- the background lives on an absolutely-positioned ::before instead, offset from
       the container's own box, since absolute left/right/top offsets aren't subject to the
       parent's flex algorithm. (min-height: 100% + flex-grow down the ancestor chain was tried
       for the same reason without a fixed number, but it pushed the scrollable chat
       history/input area out of place -- reverted.) */
    .st-key-header_banner {
        position: relative;
        border-radius: 10px;
        margin-bottom: 1rem;
        padding: 4rem 1.25rem;
        z-index: 0;
    }
    .st-key-header_banner::before {
        content: "";
        position: absolute;
        top: -96px;
        bottom: 0;
        left: -80px;
        right: -80px;
        background-color: rgba(0, 0, 0, 0.9);
        border-radius: 10px;
        z-index: -1;
    }
    [data-testid="stChatMessage"] {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border-radius: 10px;
    }
    [class*="st-key-card-"] {
        background-color: rgba(0, 0, 0, 0.6) !important;
        border-radius: 10px;
    }
    /* first child of stChatInput has no data-testid of its own -- targeted structurally
       instead of via an unstable generated CSS class */
    [data-testid="stChatInput"] > div:first-child {
        background-color: rgba(0, 0, 0, 0.6) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: white !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    /* Streamlit renders its own opaque panel around the input field (no data-testid, just a
       generated class) that otherwise sits on top of stBottom and hides its background. */
    .e15ve43o3 {
        background-color: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FREE_QUOTA = 3
EXTENDED_QUOTA = 50

# The quota is deliberately OUTSIDE st.session_state -- that resets on a plain page refresh
# (new WebSocket session = new session_state), so it would be a trivial bypass. Instead it's
# process-level shared state, keyed by _client_key() (an ID in the URL query parameter, not a
# cookie/IP/header -- those turned out unreliable in practice, see _get_or_create_visitor_id).
# Survives a refresh, but not a redeploy (acceptable for a demo; the backstop is the OpenAI
# spend cap).
#
# A plain module-level dict would NOT work -- Streamlit re-executes the whole top-level script
# on every rerun, so it would reset on every message. @st.cache_resource is what actually
# survives reruns and is shared across sessions.
_quota_lock = threading.Lock()


@st.cache_resource
def _get_quota_store() -> dict[str, int]:
    return {}


_VISITOR_ID_PARAM = "vid"

# History of failed attempts, for why this and not something "smarter":
# 1) IP/X-Forwarded-For headers -- unreliable across refreshes of the same visitor on
#    Streamlit Community Cloud.
# 2) Manual JS injection via st.iframe/document.cookie -- modern browsers partition cookies
#    set INSIDE an iframe from the main page's cookies, so they never reached the server.
# 3) streamlit-cookies-controller (a real async component) -- crashed in production with a
#    TypeError from the library's internal state, and even after wrapping it in try/except the
#    ID still changed on every refresh -- systematically unreliable on this deployment.
#
# st.query_params is purely native -- no async component, no dependency on how the
# browser/platform handles cookies. The ID is written into the URL, so a refresh loads the
# same URL including the parameter and the app sees it synchronously at script start. Cost:
# the ID is visible in the address bar (just a random UUID, nothing sensitive).
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


@st.cache_resource  # same reasoning as the quota store above -- must survive reruns
def _get_unlocked_store() -> set[str]:
    return set()


def _valid_access_codes() -> set[str]:
    """Codes live in Streamlit Secrets (same place as OPENAI_API_KEY), not in the repo."""
    try:
        return set(st.secrets.get("access_codes", {}).values())
    except Exception:
        return set()


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
    """Builds expensive shared resources once for the process lifetime, across all visitors."""
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
    st.session_state.display_log = []  # richer than conv_state.history -- also carries picks/chips for rendering
if "pending_chip" not in st.session_state:
    st.session_state.pending_chip = None


def process_message(user_message: str) -> None:
    st.session_state.display_log.append(
        {"role": "user", "content": user_message, "picks": [], "chips": []}
    )

    quota = _effective_quota()
    if _messages_used() >= quota:
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
    CHIP_RESET_ALL / CHIP_RESET_GENRE are direct commands, handled right here instead of
    going through process_message()/handle_turn(). Sending the chip text as a regular message
    and relying on keyword matching wasn't reliable -- "Zrušit omezení" (infinitive) never
    matched the check for "zruš omezení" (imperative) in brain.py. A direct command also skips
    the OpenAI call and doesn't count against the quota -- it's a UI action, not a turn.
    Returns True if chip_text was a recognized command (and handled); the caller then sends
    unrecognized text the normal way via process_message().
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

    # Persistent buttons, not just chips tied to a specific response -- those only appear on
    # the "nothing found" branch in call_llm(), so they'd disappear on a normal response.
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
        st.session_state.conv_state.sticky_constraints.clear()  # quota lives outside session_state, unaffected by this
        st.session_state.conv_state.history.clear()
        st.session_state.display_log = []
        st.rerun()

    if client is None:
        st.warning("Běží ve fallback režimu (chybí OPENAI_API_KEY) -- odpovědi nejsou generované LLM.")

with st.container(key="header_banner"):
    st.title("🎬 Movie Chatbot")
    st.caption(
        f"Doporučovací chatbot nad katalogem filmů (TMDB, {len(brain.CATALOG)} titulů). "
        "- Hybrid retrieval (tvrdý filtr → cosine similarity) "
        "- Sticky constraints extrahované klasifikátorem nad embeddingy "
        "- LLM routing mezi gpt-4o/gpt-4o-mini "
        "- Explicitní obrana proti prompt injection "
        "- Chatbot pracuje s češtinou jako primárním jazykem "
        "- Token-aware batchování přes tiktoken "
        "- nad konečným výstupem bdí Guardrails "
    )

if st.session_state.pending_chip:
    chip_text = st.session_state.pending_chip
    st.session_state.pending_chip = None
    if not _handle_chip_command(chip_text):
        process_message(chip_text)
    st.rerun()

for i, turn in enumerate(st.session_state.display_log):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

        for pick_id in turn["picks"]:
            item = catalog_by_id.get(pick_id)
            if item is None:
                continue
            with st.container(border=True, key=f"card-{i}-{pick_id}"):  # key -> targetable CSS class st-key-card-...
                st.markdown(f"**{item.title}** ({item.year}) -- _{item.genre}_")
                description = item.description
                st.caption(description[:200] + ("..." if len(description) > 200 else ""))

        if turn["chips"]:
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
