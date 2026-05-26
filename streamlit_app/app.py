import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

import asyncio

import streamlit as st

from graph.executor import GraphExecutor
from utils.state_factory import StateFactory
from utils.state_manager import StateManager
from utils.vector_store import VectorStoreManager


def run_streaming_chat_workflow(
    state,
):
    """
    Run streaming report chat workflow.
    """

    async def _stream():

        async for token in (
            st.session_state
            .graph_executor
            .stream_report_chat(
                state
            )
        ):

            yield token

    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(
        loop
    )

    async_generator = _stream()

    async def generator_wrapper():

        async for token in (
            async_generator
        ):

            yield token

    return (
        loop,
        generator_wrapper(),
    )

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

    p {
        font-size: 0.98rem !important;
        line-height: 1.75 !important;
    }

    ul {
        padding-left: 1.2rem !important;
    }

    li {
        margin-bottom: 0.45rem !important;
        line-height: 1.65 !important;
    }
    .chat-panel {
        padding: 1rem 1rem 0.75rem;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(15, 23, 42, 0.92);
        box-shadow: 0 20px 50px rgba(0,0,0,0.12);
        margin-bottom: 1rem;
    }
    .chat-panel h2 {
        font-size: 1.4rem;
        margin: 0;
        color: #f8fafc;
    }
    .chat-panel p {
        margin: 0.25rem 0 0;
        color: #cbd5e1;
        font-size: 0.96rem;
        line-height: 1.6;
    }
    section[data-testid="stChatMessageContainer"] {
        max-height: 62vh;
        min-height: 38vh;
        overflow-y: auto;
        padding: 0.8rem 0.8rem 0.5rem;
        border-radius: 1.2rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(7, 11, 24, 0.95);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04);
    }
    section[data-testid="stChatMessageContainer"]::-webkit-scrollbar {
        width: 8px;
    }
    section[data-testid="stChatMessageContainer"]::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.14);
        border-radius: 999px;
    }
    section[data-testid="stChatMessageContainer"] div[class*="stChatMessage"] {
        padding: 1rem 1.1rem;
        border-radius: 1.4rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 15px 30px rgba(0,0,0,0.14);
        background: rgba(30, 41, 59, 0.95);
        color: #e2e8f0;
        line-height: 1.78;
    }
    section[data-testid="stChatMessageContainer"] div[class*="stChatMessage"][data-author="user"] {
        background: rgba(59, 130, 246, 0.18);
        border-color: rgba(59, 130, 246, 0.28);
        color: #f8fafc;
        margin-left: auto;
    }
    section[data-testid="stChatMessageContainer"] div[class*="stChatMessage"][data-author="assistant"] {
        background: rgba(15, 23, 42, 0.95);
        border-color: rgba(148, 163, 184, 0.16);
    }
    section[data-testid="stChatMessageContainer"] div[class*="stChatMessage"] p,
    section[data-testid="stChatMessageContainer"] div[class*="stChatMessage"] span {
        color: inherit;
    }
    section[data-testid="stChatMessageContainer"] div[class*="stChatMessage"] code {
        color: #e2e8f0;
        background: rgba(148, 163, 184, 0.18);
        padding: 0.15rem 0.35rem;
        border-radius: 0.45rem;
    }
    div[data-testid="stChatInputTextArea"], textarea[data-testid="stChatInputTextArea"] {
        position: sticky !important;
        bottom: 0;
        z-index: 12;
        margin-top: 1rem;
        background: rgba(15, 23, 42, 0.96) !important;
        border-radius: 1rem !important;
        padding: 0.75rem 0.9rem !important;
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
    """
    Run the graph on the main thread's own event loop — exactly like
    the working ainvoke version, but using astream so we can write
    each agent step to an st.status() box as it arrives.

    HOW THIS WORKS
    ──────────────
    We open st.status() in a `with` block (which renders the expanding
    container in the browser immediately). Then we run a single async
    coroutine on a fresh event loop, the same way the plain ainvoke
    version does. Inside that coroutine we iterate over graph.astream()
    and call status.write() after each state update.

    Because everything runs on the main thread — no background threads,
    no queues, no polling loops — Streamlit's internal WebSocket flush
    happens normally between each await point. status.write() inside
    an active st.status context sends its delta to the browser right
    away, so the user sees each step appear as the graph progresses.

    Zero threads. Zero queues. Zero fragments. Same event-loop pattern
    as the working ainvoke version.
    """

    executor = (
        st.session_state
        .graph_executor
    )

    async def _stream():

        final_state = None

        async for event in (
            executor.stream_workflow(
                state
            )
        ):

            if (
                event["type"]
                == "event"
            ):

                status.write(
                    f"⏳ "
                    f"{event['message']}"
                )

            elif (
                event["type"]
                == "final_state"
            ):

                final_state = (
                    event["state"]
                )

        return (
            final_state
            if final_state is not None
            else state
        )

    with st.status(label, expanded=True) as status:

        loop = asyncio.new_event_loop()

        try:

            result = loop.run_until_complete(
                _stream()
            )

        except Exception as error:

            error_message = str(error)

            if (
                "rate_limit_exceeded"
                in error_message
                or "Rate limit reached"
                in error_message
                or "429"
                in error_message
            ):

                st.error(
                    "⚠️ Rate limit reached.\n\n"
                    "The language model provider "
                    "is temporarily overloaded. "
                    "Please wait a few seconds "
                    "and try again."
                )

            else:

                st.error(
                    f"Workflow Error: "
                    f"{error_message}"
                )

            result = state

        finally:

            loop.close()

        if result.get("error"):
            status.update(
                label="✗ Failed",
                state="error",
                expanded=False,
            )

        else:
            status.update(
                label="✓ Done",
                state="complete",
                expanded=False,
            )

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
        
        st.session_state["state"] = None

        st.session_state[
            "refinement_input"
        ] = ""

        st.session_state[
            "chat_history"
        ] = []

        st.session_state[
            "last_action"
        ] = ""

        st.session_state[
            "research_query"
        ] = ""

        st.session_state[
            "version_pdfs"
        ] = {}
        
        st.rerun()

    st.divider()
    st.markdown('<p class="section-label">Report History</p>', unsafe_allow_html=True)

    report_history = (
        st.session_state.state.get("report_version_history", [])
        if st.session_state.state else []
    )
    if report_history:
        for item in reversed(report_history):
            label = str(
                item.get(
                    "title",
                    item.get(
                        "description",
                        item.get(
                            "query",
                            "Unknown",
                        ),
                    ),
                )
            )
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
    st.session_state.state.get(
        "error"
    )
    if st.session_state.state
    else None
)

if workflow_error:

    clean_error = str(
        workflow_error
    )

    if (
        "rate_limit_exceeded"
        in clean_error
        or "Rate limit reached"
        in clean_error
        or "429"
        in clean_error
    ):

        st.warning(
            "⚠️ API rate limit reached.\n\n"
            "Please wait a few seconds "
            "before retrying report generation."
        )

    else:

        st.error(
            clean_error
        )

# ── Active report workspace ───────────────────────────────────────────────────
active_report = (
    st.session_state.state.get(
        "active_report",
        "",
    )
    if st.session_state.state
    else ""
)

if not isinstance(
    active_report,
    str,
):

    active_report = str(
        active_report
    )

# ── Research input ────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        "Research Topic",

        value=(
            st.session_state
            .research_query
        ),

        placeholder=(
            "e.g. Quantum Computing "
            "Applications in Drug Discovery"
        ),

        label_visibility="collapsed",

        disabled=(
            bool(active_report)
        ),
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
            st.session_state.state       = result
            st.session_state["research_query"] = query
            st.session_state.last_action = "report_generated"

        except Exception as e:
            st.error(f"Error: {e}")

if (
    st.session_state.state
    and st.session_state.state.get(
        "active_report"
    )
):
    st.divider()
    tab_report, tab_chat, tab_history = st.tabs(["📄 Report", "💬 Chat", "🕓 Versions"], key="report_workspace_tabs")

    # ── Tab 1: Report ─────────────────────────────────────────────────────────
    with tab_report:
        if st.session_state.last_action == "report_refined":
            st.success("✓ Report refined successfully.")
            st.session_state.last_action = ""

        st.markdown(
            st.session_state.state.get(
                "active_report",
                "",
            )
        )
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
                    result = run_workflow(
                        wf_state,
                        "Generating PDF…",
                    )

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
                    .get(
                        "run_metadata",
                        {},
                    )
                    .get(
                        "generated_pdf_filename",
                        "research_report.pdf",
                    )
                ),
                mime="application/pdf",
                use_container_width=True,
            )

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
        st.markdown(
            """
            <div class="chat-panel">
                <h2>Research Assistant Chat</h2>
                <p>Ask questions about your report and receive concise, professional answers with traceable context.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.chat_history:
            for chat in (
                st.session_state
                .chat_history
            ):

                with st.chat_message(
                    "user"
                ):

                    st.markdown(
                        chat["question"]
                    )

                with st.chat_message(
                    "assistant"
                ):

                    st.markdown(
                        chat["answer"]
                    )
                    
        chat_query = st.chat_input(
            "Ask questions about the report..."
        )

        if chat_query:
            if not chat_query.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    
                    with st.chat_message(
                        "user"
                    ):

                        st.markdown(
                            chat_query
                        )
                    
                    wf_state = (
                        StateManager
                        .prepare_next_workflow(
                            previous_state=(
                                st.session_state.state
                            ),

                            query=(
                                chat_query
                            ),

                            mode="REPORT_CHAT",

                            report_chat_query=(
                                chat_query
                            ),
                        )
                    )
                    
                    loop, stream = (
                        run_streaming_chat_workflow(
                            wf_state
                        )
                    )

                    response_tokens = []

                    with st.chat_message(
                        "assistant"
                    ):

                        response_container = (
                            st.empty()
                        )

                        async def consume_stream():

                            async for token in stream:

                                response_tokens.append(
                                    token
                                )

                                response_container.markdown(
                                    "".join(
                                        response_tokens
                                    )
                                )

                        try:
                            loop.run_until_complete(
                                consume_stream()
                            )

                        finally:
                            loop.close()
                    
                    full_response = "".join(
                        response_tokens
                    )
                    
                    st.session_state.chat_history.append(
                        {
                            "question": chat_query,
                            "answer": full_response,
                        }
                    )

                    wf_state[
                        "report_chat_response"
                    ] = full_response

                    st.session_state.state = (
                        wf_state
                    )

                    st.rerun()
                except Exception as e:
                    st.error(f"Chat Error: {e}")

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
                title = str(
                    version.get(
                        "title",
                        "Research Report",
                    )
                )

                timestamp = str(version.get("timestamp", ""))
                report_text = str(version.get("report", "")).strip()
                
                if not report_text:
                    st.warning(
                        "No report content found "
                        "for this version."
                    )
                    continue

                with st.expander(
                    f"{title} · Version {idx} · {timestamp}"
                ):
                    if version.get("updated_section"):
                        st.caption(f"Updated section: {version['updated_section']}")
                        st.divider()
                    st.markdown(report_text)
                    st.divider()

                    version_key = (
                        timestamp
                        .replace(":", "-")
                        .replace(" ", "_")
                    )

                    if st.button(
                        f"Generate & Download PDF — Version {timestamp}",
                        key=f"download_version_{version_key}",
                        use_container_width=True,
                    ):
                        try:
                            version_state = (
                                StateFactory
                                .create_initial_state(
                                    query=version.get(
                                        "query",
                                        "Research Report",
                                    )
                                )
                            )

                            version_state["mode"] = (
                                "DOCUMENT_GENERATION"
                            )

                            version_state["active_report"] = (
                                report_text
                            )

                            version_state["report_title"] = (
                                version.get(
                                    "title",
                                    "Research Report",
                                )
                            )

                            version_state["report_sections"] = (
                                version.get(
                                    "report_sections",
                                    {},
                                )
                            )

                            version_state["report_section_order"] = (
                                version.get(
                                    "report_section_order",
                                    [],
                                )
                            )

                            version_state["generated_pdf"] = None
                            
                            result = run_workflow(
                                version_state,
                                "Generating PDF…"
                            )

                            generated_pdf = (
                                result.get(
                                    "generated_pdf"
                                )
                            )

                            if generated_pdf:
                                version_pdfs = (
                                    st.session_state[
                                        "version_pdfs"
                                    ].copy()
                                )
                                version_pdfs[version_key] = {
                                    "pdf": generated_pdf,
                                    "title": title,
                                }
                                st.session_state[
                                    "version_pdfs"
                                ] = version_pdfs

                                st.rerun()
                            
                        except Exception as e:
                            st.error(f"Version PDF Error: {e}")
                    
                    stored_pdf = (
                        st.session_state
                        .get(
                            "version_pdfs",
                            {},
                        )
                        .get(version_key)
                    )

                    if stored_pdf:

                        st.success(
                            "✓ PDF generated successfully."
                        )

                        safe_name = (
                            stored_pdf["title"]
                            .lower()
                            .replace(" ", "_")
                            .replace("/", "_")
                        )

                        st.download_button(
                            label="⬇ Download PDF",
                            data=stored_pdf["pdf"],
                            file_name=(
                                f"{safe_name}"
                                f"_version_{idx}.pdf"
                            ),
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