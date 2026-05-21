import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(0, PROJECT_ROOT)

import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from graph.executor import GraphExecutor
from utils.state_factory import StateFactory
from utils.state_manager import StateManager
from utils.embedding_factory import EmbeddingFactory
from utils.reranker_factory import RerankerFactory

nest_asyncio.apply()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Reset & Base ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0d0f14 !important;
        color: #e8e6e0 !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        background: #0d0f14 !important;
    }

    * { font-family: 'DM Sans', sans-serif !important; }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        letter-spacing: -0.02em;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #111318 !important;
        border-right: 1px solid #1e2128 !important;
    }

    [data-testid="stSidebar"] .stButton > button {
        background: #1a1d25 !important;
        color: #c9a96e !important;
        border: 1px solid #2a2d38 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border-radius: 4px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #c9a96e !important;
        color: #0d0f14 !important;
        border-color: #c9a96e !important;
    }

    /* ── Main title ── */
    .workspace-header {
        padding: 2.5rem 0 0.5rem;
        border-bottom: 1px solid #1e2128;
        margin-bottom: 2rem;
    }

    .workspace-header h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.6rem !important;
        color: #e8e6e0 !important;
        margin: 0 !important;
    }

    .workspace-header p {
        color: #6b7280 !important;
        font-size: 0.9rem !important;
        margin: 0.4rem 0 0 !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.04em !important;
    }

    /* ── Input fields ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #111318 !important;
        border: 1px solid #1e2128 !important;
        color: #e8e6e0 !important;
        border-radius: 6px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        transition: border-color 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #c9a96e !important;
        box-shadow: 0 0 0 2px rgba(201, 169, 110, 0.12) !important;
    }

    .stTextInput label, .stTextArea label {
        color: #9ca3af !important;
        font-size: 0.8rem !important;
        font-family: 'DM Mono', monospace !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }

    /* ── Primary action button ── */
    .stButton > button[kind="primary"],
    div[data-testid="column"] .stButton > button,
    .stButton > button {
        background: #c9a96e !important;
        color: #0d0f14 !important;
        border: none !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border-radius: 4px !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #e0be87 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(201, 169, 110, 0.25) !important;
    }

    /* ── Expander toggle button — reset button styles bleeding in ── */
    [data-testid="stExpander"] summary button,
    [data-testid="stExpander"] button[kind="expanderToggle"],
    details > summary > div > button,
    details > summary {
        background: transparent !important;
        color: #9ca3af !important;
        border: none !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.82rem !important;
        font-weight: 400 !important;
        letter-spacing: 0.04em !important;
        text-transform: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid #1e2128 !important;
        gap: 0 !important;
    }

    [data-testid="stTabs"] button[role="tab"] {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #6b7280 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.75rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #c9a96e !important;
        border-bottom-color: #c9a96e !important;
    }

    [data-testid="stTabs"] button[role="tab"]:hover {
        color: #e8e6e0 !important;
    }

    /* ── Report display card ── */
    .report-card {
        background: #111318;
        border: 1px solid #1e2128;
        border-radius: 8px;
        padding: 2.5rem 3rem;
        margin: 1rem 0;
        line-height: 1.8;
        color: #e8e6e0;
    }

    .report-card h1, .report-card h2, .report-card h3 {
        color: #c9a96e !important;
        font-family: 'DM Serif Display', serif !important;
        border-bottom: 1px solid #1e2128;
        padding-bottom: 0.4rem;
        margin-top: 1.8rem;
    }

    .report-card p {
        color: #d1cfc9 !important;
        line-height: 1.85 !important;
    }

    .report-card ul, .report-card ol {
        color: #d1cfc9 !important;
    }

    .report-card code {
        font-family: 'DM Mono', monospace !important;
        background: #1a1d25 !important;
        color: #c9a96e !important;
        padding: 0.15em 0.4em !important;
        border-radius: 3px !important;
        font-size: 0.88em !important;
    }

    /* ── Alerts / Info boxes ── */
    .stAlert {
        border-radius: 6px !important;
        border-left-width: 3px !important;
        background: #111318 !important;
    }

    [data-testid="stNotification"] {
        background: #111318 !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        border: 1px solid #1e2128 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stExpander"] summary,
    .streamlit-expanderHeader {
        background: #111318 !important;
        border: none !important;
        color: #9ca3af !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stExpander"] > div:last-child,
    .streamlit-expanderContent {
        background: #111318 !important;
        border-top: 1px solid #1e2128 !important;
        padding: 1.25rem 1.5rem !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
        color: #d1cfc9 !important;
        font-size: 0.9rem !important;
        line-height: 1.8 !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h4 {
        color: #c9a96e !important;
        font-family: 'DM Serif Display', serif !important;
        border-bottom: 1px solid #1e2128 !important;
        padding-bottom: 0.3rem !important;
        margin-top: 1.4rem !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong {
        color: #e8e6e0 !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] code {
        font-family: 'DM Mono', monospace !important;
        background: #1a1d25 !important;
        color: #c9a96e !important;
        padding: 0.15em 0.4em !important;
        border-radius: 3px !important;
        font-size: 0.85em !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: transparent !important;
        color: #c9a96e !important;
        border: 1px solid #c9a96e !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }

    .stDownloadButton > button:hover {
        background: #c9a96e !important;
        color: #0d0f14 !important;
    }

    /* ── Dividers ── */
    hr {
        border-color: #1e2128 !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Sidebar history items ── */
    .history-item {
        padding: 0.75rem;
        border-left: 2px solid #c9a96e;
        margin-bottom: 0.75rem;
        background: #0d0f14;
        border-radius: 0 4px 4px 0;
    }

    .history-item h4 {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        color: #e8e6e0 !important;
        margin: 0 0 0.25rem !important;
        font-weight: 500 !important;
    }

    .history-item span {
        font-family: 'DM Mono', monospace !important;
        font-size: 0.7rem !important;
        color: #6b7280 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #c9a96e !important;
    }

    /* ── Caption text ── */
    .stCaption, caption {
        color: #6b7280 !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.04em !important;
    }

    /* ── Subheader ── */
    .stSubheader, h2 {
        color: #e8e6e0 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    /* ── Section label ── */
    .section-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #c9a96e;
        margin-bottom: 0.5rem;
    }

    /* ── Log placeholder ── */
    .workflow-log {
        background: #0a0c10;
        border: 1px solid #1e2128;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #6b7280;
        line-height: 1.7;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        color: #2a2d38;
    }

    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.4;
    }

    .empty-state p {
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        color: #3a3d48;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #0d0f14; }
    ::-webkit-scrollbar-thumb { background: #2a2d38; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #c9a96e; }

    /* Ensure report markdown renders visibly */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: #d1cfc9 !important;
    }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #c9a96e !important;
        font-family: 'DM Serif Display', serif !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: #e8e6e0 !important;
    }

    [data-testid="stMarkdownContainer"] blockquote {
        border-left: 3px solid #c9a96e !important;
        background: #111318 !important;
        padding: 0.5rem 1rem !important;
        color: #9ca3af !important;
    }

    [data-testid="stMarkdownContainer"] table {
        border-collapse: collapse !important;
        width: 100% !important;
    }

    [data-testid="stMarkdownContainer"] th {
        background: #111318 !important;
        color: #c9a96e !important;
        font-family: 'DM Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1rem !important;
        border-bottom: 1px solid #2a2d38 !important;
    }

    [data-testid="stMarkdownContainer"] td {
        padding: 0.6rem 1rem !important;
        border-bottom: 1px solid #1e2128 !important;
        color: #d1cfc9 !important;
        font-size: 0.9rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("state", None),
    ("executor", ThreadPoolExecutor(max_workers=1)),
    ("graph_executor", GraphExecutor()),
    ("refinement_input", ""),
    ("chat_input", ""),
    ("last_action", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

graph_executor = st.session_state.graph_executor


# ── Workflow helpers ──────────────────────────────────────────────────────────
async def execute_workflow_streaming(state, status_placeholder, log_placeholder):
    workflow_logs = []
    final_state = None

    async for event in graph_executor.stream_workflow(state):
        if event.get("type") == "final_state":
            final_state = event["state"]
            continue

        message = event.get("message", "Processing…")
        workflow_logs.append(message)
        status_placeholder.info(message)

        log_placeholder.markdown(
            '<div class="workflow-log">'
            + "<br>".join(f"▸ {log}" for log in workflow_logs)
            + "</div>",
            unsafe_allow_html=True,
        )

    status_placeholder.success("✓ Workflow completed")
    return final_state


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:1.5rem 0 0.5rem;">
            <span style="font-family:'DM Serif Display',serif;font-size:1.35rem;
                         color:#e8e6e0;letter-spacing:-0.02em;">Research</span>
            <span style="font-family:'DM Serif Display',serif;font-size:1.35rem;
                         color:#c9a96e;letter-spacing:-0.02em;"> Workspace</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("＋ New Research", use_container_width=True):
        for key in ("state", "refinement_input", "chat_input", "last_action"):
            st.session_state[key] = None if key == "state" else ""
        st.rerun()

    st.divider()
    st.markdown('<p class="section-label">Report History</p>', unsafe_allow_html=True)

    report_history = (
        st.session_state.state.get("report_version_history", [])
        if st.session_state.state
        else []
    )

    if report_history:
        for item in reversed(report_history):
            label = item.get("description", item.get("query", "Unknown"))
            ts = item.get("timestamp", "")
            st.markdown(
                f'<div class="history-item"><h4>{label[:40]}…</h4>'
                f"<span>{ts}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="color:#3a3d48;font-family:\'DM Mono\',monospace;'
            'font-size:0.78rem;padding:0.5rem 0;">No history yet.</p>',
            unsafe_allow_html=True,
        )


# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="workspace-header">
        <h1>AI Research Workspace</h1>
        <p>// multi-source analysis · professional reports · iterative refinement</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Research input ────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])

with col_input:
    query = st.text_input(
        "Research Topic",
        placeholder="e.g. Quantum Computing Applications in Drug Discovery",
        label_visibility="collapsed",
    )

with col_btn:
    generate_clicked = st.button("Generate →", use_container_width=True)

if generate_clicked:
    if not query.strip():
        st.warning("Please enter a research topic.")
    else:
        try:
            status_placeholder = st.empty()
            log_placeholder = st.empty()

            if st.session_state.state is None:
                state = StateFactory.create_initial_state(query=query)
            else:
                state = StateManager.prepare_next_workflow(
                    previous_state=st.session_state.state,
                    query=query,
                    mode="REPORT_GENERATION",
                )

            with st.spinner("Generating report…"):
                result = asyncio.run(
                    execute_workflow_streaming(
                        state=state,
                        status_placeholder=status_placeholder,
                        log_placeholder=log_placeholder,
                    )
                )

            st.session_state.state = result
            st.session_state.last_action = "report_generated"
            st.rerun()

        except Exception as error:
            st.error(f"Error: {error}")


# ── Active report workspace ───────────────────────────────────────────────────
active_report = (
    st.session_state.state.get("active_report", "")
    if st.session_state.state
    else ""
)

if st.session_state.state and active_report:
    st.divider()

    tab_report, tab_chat, tab_history = st.tabs(
        ["📄 Report", "💬 Chat", "🕓 Versions"]
    )

    # ── Tab 1: Report ─────────────────────────────────────────────────────────
    with tab_report:

        if st.session_state.last_action == "report_refined":
            st.success("✓ Report refined successfully.")
            st.session_state.last_action = ""

        # ── Report rendered in styled card ──
        st.markdown(
            '<p class="section-label">Generated Report</p>',
            unsafe_allow_html=True,
        )

        # Wrap report in styled card container
        st.markdown(
            f'<div class="report-card">{active_report}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Export ──
        st.markdown('<p class="section-label">Export</p>', unsafe_allow_html=True)

        col_info, col_gen = st.columns([3, 1])
        with col_info:
            st.caption("Review the report above before exporting as PDF.")
        with col_gen:
            if st.button("Generate PDF", use_container_width=True):
                try:
                    status_placeholder = st.empty()
                    log_placeholder = st.empty()

                    state = StateManager.prepare_next_workflow(
                        previous_state=st.session_state.state,
                        query=st.session_state.state["query"],
                        mode="DOCUMENT_GENERATION",
                    )

                    with st.spinner("Generating PDF…"):
                        result = asyncio.run(
                            execute_workflow_streaming(
                                state=state,
                                status_placeholder=status_placeholder,
                                log_placeholder=log_placeholder,
                            )
                        )
                    st.session_state.state = result
                    st.rerun()

                except Exception as error:
                    st.error(f"PDF Error: {error}")

        generated_pdf = st.session_state.state.get("generated_pdf")
        if generated_pdf:
            st.success("✓ PDF ready.")
            st.download_button(
                label="⬇ Download PDF",
                data=generated_pdf,
                file_name="research_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.divider()

        # ── Refine ──
        st.markdown(
            '<p class="section-label">Refine Report</p>',
            unsafe_allow_html=True,
        )

        refinement_query = st.text_area(
            "Refinement Instructions",
            value=st.session_state.refinement_input,
            placeholder=(
                "e.g. Add recent industry applications, security challenges, "
                "and future trends. Expand the executive summary."
            ),
            height=120,
            label_visibility="collapsed",
        )

        if st.button("Apply Refinement →", use_container_width=True):
            if not refinement_query.strip():
                st.warning("Please enter refinement instructions.")
            else:
                try:
                    status_placeholder = st.empty()
                    log_placeholder = st.empty()

                    state = StateManager.prepare_next_workflow(
                        previous_state=st.session_state.state,
                        query=refinement_query,
                        mode="REPORT_REFINEMENT",
                        refinement_query=refinement_query,
                    )

                    with st.spinner("Refining report…"):
                        result = asyncio.run(
                            execute_workflow_streaming(
                                state=state,
                                status_placeholder=status_placeholder,
                                log_placeholder=log_placeholder,
                            )
                        )

                    st.session_state.state = result
                    st.session_state.refinement_input = ""
                    st.session_state.last_action = "report_refined"
                    st.rerun()

                except Exception as error:
                    st.error(f"Refinement Error: {error}")

    # ── Tab 2: Chat ───────────────────────────────────────────────────────────
    with tab_chat:
        st.markdown(
            '<p class="section-label">Ask Questions About This Report</p>',
            unsafe_allow_html=True,
        )

        col_q, col_ask = st.columns([5, 1])
        with col_q:
            chat_query = st.text_input(
                "Question",
                value=st.session_state.chat_input,
                placeholder="e.g. What were the main technical findings?",
                label_visibility="collapsed",
            )
        with col_ask:
            ask_clicked = st.button("Ask →", use_container_width=True)

        if ask_clicked:
            if not chat_query.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    status_placeholder = st.empty()
                    log_placeholder = st.empty()

                    state = StateManager.prepare_next_workflow(
                        previous_state=st.session_state.state,
                        query=chat_query,
                        mode="REPORT_CHAT",
                        report_chat_query=chat_query,
                    )

                    with st.spinner("Thinking…"):
                        result = asyncio.run(
                            execute_workflow_streaming(
                                state=state,
                                status_placeholder=status_placeholder,
                                log_placeholder=log_placeholder,
                            )
                        )

                    st.session_state.state = result
                    st.session_state.chat_input = ""
                    st.rerun()

                except Exception as error:
                    st.error(f"Chat Error: {error}")

        report_chat_response = st.session_state.state.get("report_chat_response", "")
        if report_chat_response:
            st.divider()
            st.markdown('<p class="section-label">Answer</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="report-card" style="border-left:3px solid #c9a96e;">'
                f"{report_chat_response}</div>",
                unsafe_allow_html=True,
            )

    # ── Tab 3: Version History ────────────────────────────────────────────────
    with tab_history:
        version_history = st.session_state.state.get("report_version_history", [])

        if not version_history:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-state-icon">◎</div>'
                "<p>No versions yet — generate or refine a report to begin.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="section-label">Version History</p>',
                unsafe_allow_html=True,
            )
            st.caption(f"{len(version_history)} version(s) saved")

            for idx, version in enumerate(reversed(version_history), start=1):
                desc = version.get("description", version.get("mode", "Unknown"))
                ts = version.get("timestamp", "")

                with st.expander(f"Version {idx}  ·  {desc}  ·  {ts}"):
                    if version.get("updated_section"):
                        st.caption(f"Updated section: {version['updated_section']}")

                    report_text = version.get("report", "")
                    st.markdown(report_text)

else:
    # ── Empty state when no report ────────────────────────────────────────────
    if not st.session_state.state:
        st.markdown(
            """
            <div class="empty-state" style="margin-top:4rem;">
                <div class="empty-state-icon">⬡</div>
                <p>Enter a topic above and press Generate to begin your research.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )