import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

import asyncio

import streamlit as st
from datetime import datetime

from graph.executor import GraphExecutor
from utils.state_factory import StateFactory
from utils.state_manager import StateManager
from utils.vector_store import VectorStoreManager


def run_streaming_chat_workflow(state):
    """Run streaming report chat workflow."""

    async def _stream():
        async for token in (
            st.session_state.graph_executor.stream_report_chat(state)
        ):
            yield token

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_generator = _stream()

    async def generator_wrapper():
        async for token in async_generator:
            yield token

    return loop, generator_wrapper()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .section-label {
        font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase;
        color: #888; margin-bottom: 0.4rem; font-weight: 600;
    }
    .history-item {
        padding: 0.6rem 0.75rem; border-left: 3px solid #f63366;
        margin-bottom: 0.6rem; border-radius: 0 4px 4px 0;
        background: rgba(246,51,102,0.04);
    }
    .history-item h4 { font-size: 0.85rem; margin: 0 0 0.2rem; font-weight: 500; }
    .history-item span { font-size: 0.72rem; color: #888; font-family: monospace; }
    .empty-state { text-align: center; padding: 5rem 2rem; }
    .empty-state-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.4; }
    .empty-state p { font-size: 0.88rem; color: #aaa; }

    h1 {
        font-size: 2.7rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 2rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.03em !important;
    }
    h2 {
        font-size: 1.55rem !important;
        margin-top: 2.2rem !important;
        margin-bottom: 1rem !important;
        font-weight: 700 !important;
        padding-bottom: 0.35rem !important;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    h3 {
        font-size: 1rem !important;
        margin-top: 1.3rem !important;
        margin-bottom: 0.45rem !important;
        font-weight: 600 !important;
        color: #d0d0d0 !important;
        opacity: 0.92;
    }
    p  { font-size: 0.98rem !important; line-height: 1.75 !important; }
    ul { padding-left: 1.2rem !important; }
    li { margin-bottom: 0.45rem !important; line-height: 1.65 !important; }

    /* ── Chat interface ─────────────────────────────────────────────────── */
    .chat-shell {
        display: flex;
        flex-direction: column;
        height: 72vh;
        border: 0.5px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        overflow: hidden;
        background: rgba(10, 15, 30, 0.96);
    }
    .chat-topbar {
        flex-shrink: 0;
        padding: 0.75rem 1.1rem;
        border-bottom: 0.5px solid rgba(255,255,255,0.08);
        background: rgba(15, 22, 42, 0.98);
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .chat-topbar-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #22c55e;
        flex-shrink: 0;
    }
    .chat-topbar-title {
        font-size: 0.88rem;
        font-weight: 500;
        color: #e2e8f0;
        margin: 0;
        line-height: 1;
    }
    .chat-topbar-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin: 0;
        line-height: 1;
    }
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 1rem 1.1rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        scroll-behavior: smooth;
    }
    .chat-messages::-webkit-scrollbar { width: 4px; }
    .chat-messages::-webkit-scrollbar-track { background: transparent; }
    .chat-messages::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.10);
        border-radius: 99px;
    }
    .chat-empty-state {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        color: #475569;
        font-size: 0.85rem;
        padding: 2rem;
    }
    .chat-empty-icon {
        font-size: 1.8rem;
        opacity: 0.35;
        margin-bottom: 0.25rem;
    }
    .msg-row {
        display: flex;
        gap: 0.6rem;
        align-items: flex-end;
    }
    .msg-row.user { flex-direction: row-reverse; }
    .msg-row.assistant { flex-direction: row; }
    .msg-avatar {
        width: 28px; height: 28px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.02em;
    }
    .msg-avatar.user-av {
        background: rgba(59,130,246,0.22);
        color: #93c5fd;
        border: 0.5px solid rgba(59,130,246,0.25);
    }
    .msg-avatar.asst-av {
        background: rgba(168,85,247,0.18);
        color: #c4b5fd;
        border: 0.5px solid rgba(168,85,247,0.22);
    }
    .msg-body { display: flex; flex-direction: column; max-width: 75%; gap: 3px; }
    .msg-row.user  .msg-body { align-items: flex-end; }
    .msg-row.assistant .msg-body { align-items: flex-start; }
    .msg-bubble {
        padding: 0.55rem 0.85rem;
        font-size: 0.875rem;
        line-height: 1.65;
        word-break: break-word;
    }
    .msg-bubble.user-bubble {
        background: rgba(37,99,235,0.28);
        border: 0.5px solid rgba(59,130,246,0.30);
        border-radius: 16px 16px 4px 16px;
        color: #e0eeff;
    }
    .msg-bubble.asst-bubble {
        background: rgba(30, 38, 60, 0.90);
        border: 0.5px solid rgba(255,255,255,0.08);
        border-radius: 16px 16px 16px 4px;
        color: #cbd5e1;
    }
    .msg-meta {
        font-size: 0.68rem;
        color: #475569;
        padding: 0 4px;
    }
    .chat-input-bar {
        flex-shrink: 0;
        border-top: 0.5px solid rgba(255,255,255,0.07);
        background: rgba(15, 22, 42, 0.98);
        padding: 0.6rem 0.8rem;
    }
    /* strip Streamlit's default chat_input decoration inside the shell */
    .chat-input-bar section[data-testid="stChatInputContainer"],
    .chat-input-bar div[data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    /* override global stChatMessageContainer if it leaks */
    section[data-testid="stChatMessageContainer"] {
        max-height: unset !important;
        min-height: unset !important;
        overflow: visible !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        border-radius: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Agent step labels ─────────────────────────────────────────────────────────
AGENT_MESSAGES = {
    "supervisor_agent":          "Supervisor evaluating workflow…",
    "context_agent":             "Processing contextual query…",
    "query_decomposition_agent": "Decomposing research query…",
    "retrieval_agent":           "Retrieving research documents…",
    "analysis_agent":            "Analyzing retrieved information…",
    "report_generation_agent":   "Generating research report…",
    "report_refinement_agent":   "Refining report sections…",
    "report_chat_agent":         "Generating conversational response…",
    "document_generation_agent": "Generating PDF document…",
    "error_recovery_agent":      "Handling workflow error…",
}

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("state",            None),
    ("graph_executor",   None),
    ("refinement_input", ""),
    ("chat_input",       ""),
    ("chat_history",     []),
    ("last_action",      ""),
    ("research_query",   ""),
    ("version_pdfs",     {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.graph_executor is None:
    st.session_state.graph_executor = GraphExecutor()


# ── Workflow runner ───────────────────────────────────────────────────────────
def run_workflow(state, label: str = "Running workflow…"):
    executor = st.session_state.graph_executor

    async def _stream():
        final_state = None
        async for event in executor.stream_workflow(state):
            if event["type"] == "event":
                status.write(f"⏳ {event['message']}")
            elif event["type"] == "final_state":
                final_state = event["state"]
        return final_state if final_state is not None else state

    with st.status(label, expanded=True) as status:
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_stream())
        except Exception as error:
            error_message = str(error)
            if (
                "rate_limit_exceeded" in error_message
                or "Rate limit reached" in error_message
                or "429" in error_message
            ):
                st.error(
                    "⚠️ Rate limit reached.\n\n"
                    "The language model provider is temporarily overloaded. "
                    "Please wait a few seconds and try again."
                )
            else:
                st.error(f"Workflow Error: {error_message}")
            result = state
        finally:
            loop.close()

        if result.get("error"):
            status.update(label="✗ Failed", state="error", expanded=False)
        else:
            status.update(label="✓ Done", state="complete", expanded=False)

    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Research Workspace")
    st.divider()

    if st.button("＋ New Research", use_container_width=True):
        try:
            VectorStoreManager.clear_persisted_data()
        except Exception:
            pass

        st.session_state["state"]            = None
        st.session_state["refinement_input"] = ""
        st.session_state["chat_history"]     = []
        st.session_state["last_action"]      = ""
        st.session_state["research_query"]   = ""
        st.session_state["version_pdfs"]     = {}
        st.rerun()

    st.divider()
    st.markdown('<p class="section-label">Report History</p>', unsafe_allow_html=True)

    report_history = (
        st.session_state.state.get("report_version_history", [])
        if st.session_state.state else []
    )
    if report_history:
        for item in reversed(report_history):
            label = str(item.get("title", item.get("description", item.get("query", "Unknown"))))
            ts    = str(item.get("timestamp", ""))
            st.markdown(
                f'<div class="history-item"><h4>{label[:40]}…</h4>'
                f"<span>{ts}</span></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No history yet.")


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("AI Research Workspace")
st.caption("Multi-source analysis · Professional reports · Iterative refinement")
st.divider()

workflow_error = (
    st.session_state.state.get("error")
    if st.session_state.state else None
)

if workflow_error:
    clean_error = str(workflow_error)
    if (
        "rate_limit_exceeded" in clean_error
        or "Rate limit reached" in clean_error
        or "429" in clean_error
    ):
        st.warning(
            "⚠️ API rate limit reached.\n\n"
            "Please wait a few seconds before retrying report generation."
        )
    else:
        st.error(clean_error)

# ── Active report ─────────────────────────────────────────────────────────────
active_report = (
    st.session_state.state.get("active_report", "")
    if st.session_state.state else ""
)
if not isinstance(active_report, str):
    active_report = str(active_report)

# ── Research input ────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Research Topic",
        value=st.session_state.research_query,
        placeholder="e.g. Quantum Computing Applications in Drug Discovery",
        label_visibility="collapsed",
        disabled=bool(active_report),
    )
with col_btn:
    generate_clicked = st.button(
        "Generate →",
        use_container_width=True,
        disabled=bool(active_report),
    )

if generate_clicked:
    if not query.strip():
        st.warning("Please enter a research topic.")
    else:
        try:
            if st.session_state.state is None:
                wf_state = StateFactory.create_initial_state(query=query)
            else:
                wf_state = StateManager.prepare_next_workflow(
                    previous_state=st.session_state.state,
                    query=query,
                    mode="REPORT_GENERATION",
                )
            result = run_workflow(wf_state, "Generating report…")
            st.session_state.state         = result
            st.session_state.research_query = query
            st.session_state.last_action   = "report_generated"
        except Exception as e:
            st.error(f"Error: {e}")


# ── Workspace tabs ────────────────────────────────────────────────────────────
if st.session_state.state and st.session_state.state.get("active_report"):
    st.divider()
    tab_report, tab_chat, tab_history = st.tabs(
        ["📄 Report", "💬 Chat", "🕓 Versions"],
        key="report_workspace_tabs",
    )

    # ── Tab 1: Report ─────────────────────────────────────────────────────────
    with tab_report:
        if st.session_state.last_action == "report_refined":
            st.success("✓ Report refined successfully.")
            st.session_state.last_action = ""

        st.markdown(st.session_state.state.get("active_report", ""))
        st.divider()

        # Export
        st.markdown('<p class="section-label">Export</p>', unsafe_allow_html=True)
        col_info, col_gen = st.columns([3, 1])
        with col_info:
            st.caption("Review the report above before exporting as PDF.")
        with col_gen:
            if st.button("Generate PDF", use_container_width=True):
                try:
                    wf_state = StateManager.prepare_next_workflow(
                        previous_state=st.session_state.state,
                        query=st.session_state.state["query"],
                        mode="DOCUMENT_GENERATION",
                    )
                    result = run_workflow(wf_state, "Generating PDF…")
                    if not result.get("error"):
                        st.session_state.state = result
                except Exception as e:
                    st.error(f"PDF Error: {e}")

        generated_pdf = st.session_state.state.get("generated_pdf")
        if generated_pdf:
            st.success("✓ PDF ready.")
            st.download_button(
                label="⬇ Download PDF",
                data=generated_pdf,
                file_name=(
                    st.session_state.state
                    .get("run_metadata", {})
                    .get("generated_pdf_filename", "research_report.pdf")
                ),
                mime="application/pdf",
                use_container_width=True,
            )

        # Regenerate: restart the whole report workflow from the beginning
        if st.button("Regenerate Report", use_container_width=True):
            # Prevent duplicate handling if a regenerate is already running
            if st.session_state.get("regen_in_progress"):
                st.warning("Regeneration already in progress.")
            else:
                st.session_state["regen_in_progress"] = True
                try:
                    query_for_regen = (
                        st.session_state.state.get("query")
                        or st.session_state.research_query
                        or ""
                    )
                    if not query_for_regen:
                        st.warning("Original query missing; cannot regenerate.")
                    else:
                        # store current active report as a version before regenerating
                        prev_state = st.session_state.state or {}
                        active = prev_state.get("active_report", "")

                        # capture previous version history (may be empty)
                        version_history = prev_state.get("report_version_history", []).copy()

                        if active:
                            version = {
                                "title": prev_state.get("report_title") or prev_state.get("run_metadata", {}).get("generated_pdf_filename", "Research Report"),
                                "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
                                "report": active,
                                "query": prev_state.get("query", query_for_regen),
                                "report_sections": prev_state.get("report_sections", {}),
                                "report_section_order": prev_state.get("report_section_order", []),
                                "updated_section": prev_state.get("updated_section"),
                            }
                            # avoid appending duplicate of the most recent saved version
                            if not version_history or version_history[-1].get("report") != version.get("report"):
                                version_history.append(version)

                        # run the regeneration workflow
                        wf_state = StateFactory.create_initial_state(query=query_for_regen)
                        # when a user explicitly requests regeneration, disable automatic
                        # retrieval retry attempts to avoid prolonged retry loops
                        wf_state["max_retries"] = 0
                        wf_state.setdefault("run_metadata", {})["regen_initiated"] = True
                        result = run_workflow(wf_state, "Regenerating report…")

                        # ensure the new state preserves the version history we saved
                        if result is None:
                            result = {}

                        # merge any existing versions from result with our saved history, preferring saved history first
                        new_history = version_history + result.get("report_version_history", [])

                        # deduplicate consecutive identical entries
                        deduped = []
                        for v in new_history:
                            if not deduped or deduped[-1].get("report") != v.get("report"):
                                deduped.append(v)

                        result["report_version_history"] = deduped

                        st.session_state.state = result
                        st.session_state.research_query = query_for_regen
                        st.session_state.last_action = "report_generated"
                        st.rerun()
                except Exception as e:
                    st.error(f"Regenerate Error: {e}")
                finally:
                    st.session_state["regen_in_progress"] = False

        st.divider()

        # Refine
        st.markdown('<p class="section-label">Refine Report</p>', unsafe_allow_html=True)
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
                    wf_state = StateManager.prepare_next_workflow(
                        previous_state=st.session_state.state,
                        query=refinement_query,
                        mode="REPORT_REFINEMENT",
                        refinement_query=refinement_query,
                    )
                    result = run_workflow(wf_state, "Refining report…")
                    st.session_state.state            = result
                    st.session_state.refinement_input = ""
                    st.session_state.last_action      = "report_refined"
                    st.rerun()
                except Exception as e:
                    st.error(f"Refinement Error: {e}")

    # ── Tab 2: Chat ───────────────────────────────────────────────────────────
    with tab_chat:

        # ── Top bar ───────────────────────────────────────────────────────────────
        st.markdown("""
        <div style="
            display:flex; align-items:center; gap:8px;
            padding:10px 14px;
            border:0.5px solid rgba(255,255,255,0.08);
            border-radius:12px 12px 0 0;
            background:rgba(15,22,42,0.98);
            margin-bottom:0;
        ">
            <div style="width:7px;height:7px;border-radius:50%;background:#22c55e;flex-shrink:0;"></div>
            <div>
                <p style="font-size:0.82rem;font-weight:500;color:#e2e8f0;margin:0;line-height:1;">Report assistant</p>
                <p style="font-size:0.7rem;color:#64748b;margin:0;line-height:1;">Ask anything about your report</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Message history ───────────────────────────────────────────────────────
        chat_container = st.container(height=420, border=True)

        with chat_container:
            if not st.session_state.chat_history:
                st.markdown(
                    "<div style='text-align:center;padding:3rem 1rem;color:#475569;font-size:0.82rem;'>"
                    "<div style='font-size:1.6rem;opacity:0.3;margin-bottom:0.5rem;'>💬</div>"
                    "No messages yet — ask a question below."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                for chat in st.session_state.chat_history:
                    with st.chat_message("user"):
                        st.write(chat["question"])
                    with st.chat_message("assistant"):
                        st.write(chat["answer"])

        # ── Input ─────────────────────────────────────────────────────────────────
        chat_query = st.chat_input("Ask about the report…", key="report_chat_input")

        # ── Handle submission ─────────────────────────────────────────────────────
        if chat_query and chat_query.strip():

            # 1. Show the user message immediately (optimistic)
            with chat_container:
                with st.chat_message("user"):
                    st.write(chat_query)
                response_placeholder = st.chat_message("assistant")

            # 2. Stream the response into the placeholder
            try:
                wf_state = StateManager.prepare_next_workflow(
                    previous_state=st.session_state.state,
                    query=chat_query,
                    mode="REPORT_CHAT",
                    report_chat_query=chat_query,
                )

                loop, stream = run_streaming_chat_workflow(wf_state)
                response_tokens = []

                with response_placeholder:
                    stream_display = st.empty()

                    async def consume_stream():
                        async for token in stream:
                            response_tokens.append(token)
                            stream_display.write("".join(response_tokens) + "▌")

                    try:
                        loop.run_until_complete(consume_stream())
                    finally:
                        loop.close()

                full_response = "".join(response_tokens)

                # 3. Persist and rerun so the history container rerenders cleanly
                st.session_state.chat_history.append({
                    "question": chat_query,
                    "answer":   full_response,
                })
                wf_state["report_chat_response"] = full_response
                st.session_state.state = wf_state
                st.rerun()

            except Exception as e:
                st.error(f"Chat error: {e}")

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
            st.markdown('<p class="section-label">Version History</p>', unsafe_allow_html=True)
            st.caption(f"{len(version_history)} version(s) saved")

            for idx, version in enumerate(reversed(version_history), start=1):
                title       = str(version.get("title", "Research Report"))
                timestamp   = str(version.get("timestamp", ""))
                report_text = str(version.get("report", "")).strip()

                if not report_text:
                    st.warning("No report content found for this version.")
                    continue

                with st.expander(f"{title} · Version {idx} · {timestamp}"):
                    if version.get("updated_section"):
                        st.caption(f"Updated section: {version['updated_section']}")
                        st.divider()
                    st.markdown(report_text)
                    st.divider()

                    version_key = (
                        timestamp.replace(":", "-").replace(" ", "_")
                    )

                    if st.button(
                        f"Generate & Download PDF — Version {timestamp}",
                        key=f"download_version_{version_key}",
                        use_container_width=True,
                    ):
                        try:
                            version_state = StateFactory.create_initial_state(
                                query=version.get("query", "Research Report")
                            )
                            version_state["mode"]                 = "DOCUMENT_GENERATION"
                            version_state["active_report"]        = report_text
                            version_state["report_title"]         = version.get("title", "Research Report")
                            version_state["report_sections"]      = version.get("report_sections", {})
                            version_state["report_section_order"] = version.get("report_section_order", [])
                            version_state["generated_pdf"]        = None

                            result = run_workflow(version_state, "Generating PDF…")

                            generated_pdf = result.get("generated_pdf")
                            if generated_pdf:
                                version_pdfs = st.session_state["version_pdfs"].copy()
                                version_pdfs[version_key] = {
                                    "pdf":   generated_pdf,
                                    "title": title,
                                }
                                st.session_state["version_pdfs"] = version_pdfs
                                st.rerun()

                        except Exception as e:
                            st.error(f"Version PDF Error: {e}")

                    stored_pdf = st.session_state.get("version_pdfs", {}).get(version_key)
                    if stored_pdf:
                        st.success("✓ PDF generated successfully.")
                        safe_name = (
                            stored_pdf["title"]
                            .lower()
                            .replace(" ", "_")
                            .replace("/", "_")
                        )
                        st.download_button(
                            label="⬇ Download PDF",
                            data=stored_pdf["pdf"],
                            file_name=f"{safe_name}_version_{idx}.pdf",
                            mime="application/pdf",
                            key=f"pdf_download_{version_key}",
                            use_container_width=True,
                        )

else:
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