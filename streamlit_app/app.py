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
    ("last_action",      ""),
    ("research_query",   ""),
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

        return final_state

    with st.status(label, expanded=True) as status:

        loop = asyncio.new_event_loop()

        try:

            result = loop.run_until_complete(
                _stream()
            )

        except Exception as error:

            st.error(
                f"Workflow Error: {error}"
            )

            result = state

        finally:

            loop.close()

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
        for key in (
            "state",
            "refinement_input",
            "chat_input",
            "last_action",
            "research_query",
        ):
            st.session_state[key] = None if key == "state" else ""
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
    generate_clicked = st.button("Generate →", use_container_width=True)

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
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.state and active_report:
    st.divider()
    tab_report, tab_chat, tab_history = st.tabs(["📄 Report", "💬 Chat", "🕓 Versions"])

    # ── Tab 1: Report ─────────────────────────────────────────────────────────
    with tab_report:
        if st.session_state.last_action == "report_refined":
            st.success("✓ Report refined successfully.")
            st.session_state.last_action = ""

        st.markdown(
            active_report,
            unsafe_allow_html=True,
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
                    result = run_workflow(wf_state, "Generating PDF…")
                    st.session_state.state = result
                    st.rerun()
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
        st.markdown('<p class="section-label">Ask Questions About This Report</p>', unsafe_allow_html=True)
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
                    wf_state = StateManager.prepare_next_workflow(
                        previous_state=st.session_state.state,
                        query=chat_query,
                        mode="REPORT_CHAT",
                        report_chat_query=chat_query,
                    )
                    result = run_workflow(wf_state, "Thinking…")
                    st.session_state.state      = result
                    st.session_state.chat_input = ""
                    st.rerun()
                except Exception as e:
                    st.error(f"Chat Error: {e}")

        report_chat_response = (
            st.session_state.state.get(
                "report_chat_response",
                "",
            )
            if st.session_state.state
            else ""
        )

        if not isinstance(
            report_chat_response,
            str,
        ):

            report_chat_response = str(
                report_chat_response
            )
            
        if report_chat_response:
            st.divider()
            st.markdown('<p class="section-label">Answer</p>', unsafe_allow_html=True)
            st.info(report_chat_response)

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

                desc = str(
                    version.get(
                        "description",
                        version.get(
                            "mode",
                            "Unknown",
                        ),
                    )
                )
                ts          = str(version.get("timestamp", ""))
                report_text = str(version.get("report", ""))

                with st.expander(
                    f"{title} · Version {idx} · {ts}"
                ):
                    if version.get("updated_section"):
                        st.caption(f"Updated section: {version['updated_section']}")
                        st.divider()
                    st.markdown(report_text)
                    st.divider()

                    if st.button(
                        f"Generate & Download PDF — Version {idx}",
                        key=f"download_version_{idx}",
                        use_container_width=True,
                    ):
                        try:
                            temp_state = dict(st.session_state.state)
                            temp_state["active_report"] = report_text
                            temp_state[
                                "report_title"
                            ] = version.get(
                                "title",
                                temp_state.get(
                                    "report_title",
                                    temp_state.get(
                                        "query",
                                        "Research Report",
                                    ),
                                ),
                            )
                            wf_state = StateManager.prepare_next_workflow(
                                previous_state=temp_state,
                                query=temp_state["query"],
                                mode="DOCUMENT_GENERATION",
                            )
                            result = run_workflow(wf_state, "Generating PDF…")
                            generated_pdf = result.get("generated_pdf")
                            if generated_pdf:
                                st.success("✓ PDF generated successfully.")
                                safe_name = (
                                    title
                                    .lower()
                                    .replace(" ", "_")
                                    .replace("/", "_")
                                )
                                st.download_button(
                                    label="⬇ Download PDF",
                                    data=generated_pdf,
                                    file_name=f"{safe_name}_version_{idx}.pdf",
                                    mime="application/pdf",
                                    key=f"pdf_download_{idx}",
                                    use_container_width=True,
                                )
                        except Exception as e:
                            st.error(f"Version PDF Error: {e}")

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