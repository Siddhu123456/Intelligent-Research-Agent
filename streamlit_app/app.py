import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT,
)

import asyncio
import queue
import threading

from streamlit.runtime.scriptrunner import (
    get_script_run_ctx,
    add_script_run_ctx,
)

import streamlit as st

from graph.executor import (
    GraphExecutor,
)

from utils.state_factory import (
    StateFactory,
)

from utils.state_manager import (
    StateManager,
)

from utils.vector_store import (
    VectorStoreManager,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=(
        "AI Research Workspace"
    ),
    layout="wide",
    initial_sidebar_state=(
        "expanded"
    ),
)

# ── Safe helpers ─────────────────────────────────────────────────────────────

def safe_text(
    value,
) -> str:

    if value is None:

        return ""

    if isinstance(
        value,
        dict,
    ):

        for key in [
            "report",
            "response",
            "content",
            "text",
            "refined_report",
        ]:

            if key in value:

                return str(
                    value[key]
                )

    return str(
        value
    )


# ── Minimal CSS ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>

    .section-label {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }

    .history-item {
        padding: 0.6rem 0.75rem;
        border-left: 3px solid #f63366;
        margin-bottom: 0.6rem;
        border-radius: 0 4px 4px 0;
        background: rgba(246, 51, 102, 0.04);
    }

    .history-item h4 {
        font-size: 0.85rem;
        margin: 0 0 0.2rem;
        font-weight: 500;
    }

    .history-item span {
        font-size: 0.72rem;
        color: #888;
        font-family: monospace;
    }

    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
    }

    .empty-state-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        opacity: 0.4;
    }

    .empty-state p {
        font-size: 0.88rem;
        color: #aaa;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ── Agent step labels ────────────────────────────────────────────────────────

AGENT_MESSAGES = {
    "supervisor_agent":
        "Supervisor evaluating workflow…",

    "context_agent":
        "Processing contextual query…",

    "query_decomposition_agent":
        "Decomposing research query…",

    "retrieval_agent":
        "Retrieving research documents…",

    "analysis_agent":
        "Analyzing retrieved information…",

    "report_generation_agent":
        "Generating research report…",

    "report_refinement_agent":
        "Refining report sections…",

    "report_chat_agent":
        "Generating conversational response…",

    "document_generation_agent":
        "Generating PDF document…",

    "error_recovery_agent":
        "Handling workflow error…",
}

# ── Session state defaults ──────────────────────────────────────────────────

for key, default in [
    ("state", None),
    ("graph_executor", None),
    ("refinement_input", ""),
    ("chat_input", ""),
    ("last_action", ""),
]:

    if key not in st.session_state:

        st.session_state[key] = (
            default
        )

if (
    st.session_state.graph_executor
    is None
):

    st.session_state.graph_executor = (
        GraphExecutor()
    )

# ── Workflow runner ──────────────────────────────────────────────────────────

def run_async_workflow(
    state,
):

    ctx = (
        get_script_run_ctx()
    )

    graph = (
        st.session_state
        .graph_executor
        ._graph
    )

    msg_queue: queue.Queue = (
        queue.Queue()
    )

    done_event = (
        threading.Event()
    )

    result_holder = [None]

    error_holder = [None]

    async def _run():

        final_state = None

        previous_agent = None

        async for state_update in graph.astream(
            state,
            stream_mode="values",
        ):

            final_state = (
                state_update
            )

            current_agent = (
                state_update.get(
                    "current_agent",
                    "",
                )
            )

            if (
                current_agent
                and current_agent
                != previous_agent
            ):

                previous_agent = (
                    current_agent
                )

                label = (
                    AGENT_MESSAGES.get(
                        current_agent,
                        f"{current_agent}…",
                    )
                )

                msg_queue.put(
                    label
                )

        return final_state

    def worker():

        add_script_run_ctx(
            threading.current_thread(),
            ctx,
        )

        loop = (
            asyncio.new_event_loop()
        )

        asyncio.set_event_loop(
            loop
        )

        try:

            result_holder[0] = (
                loop.run_until_complete(
                    _run()
                )
            )

        except Exception as exc:

            error_holder[0] = exc

        finally:

            loop.close()

            done_event.set()

    thread = threading.Thread(
        target=worker,
        daemon=True,
    )

    thread.start()

    log_placeholder = (
        st.empty()
    )

    steps: list[str] = []

    def _render(
        done: bool = False,
    ):

        lines = [
            f"✔ {s}"
            for s in steps[:-1]
        ]

        if steps:

            last = steps[-1]

            lines.append(
                (
                    f"✔ {last}"
                    if done
                    else f"⏳ {last}"
                )
            )

        if done:

            lines.append(
                "**✓ Done**"
            )

        log_placeholder.markdown(
            "\n\n".join(lines)
            if lines
            else "_Starting…_"
        )

    _render()

    while not done_event.is_set():

        try:

            msg = msg_queue.get(
                timeout=0.1
            )

            steps.append(
                msg
            )

            _render()

        except queue.Empty:

            continue

    while not msg_queue.empty():

        try:

            steps.append(
                msg_queue.get_nowait()
            )

        except queue.Empty:

            break

    _render(done=True)

    thread.join()

    if error_holder[0]:

        raise error_holder[0]

    return result_holder[0]


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:

    st.title(
        "Research Workspace"
    )

    st.divider()

    if st.button(
        "＋ New Research",
        use_container_width=True,
    ):

        try:

            VectorStoreManager.clear_persisted_data()

        except Exception:

            pass

        for key in [
            "state",
            "refinement_input",
            "chat_input",
            "last_action",
        ]:

            st.session_state[key] = (
                None
                if key == "state"
                else ""
            )

        st.rerun()

    st.divider()

    st.markdown(
        '<p class="section-label">'
        'Report History'
        '</p>',
        unsafe_allow_html=True,
    )

    report_history = []

    if st.session_state.state:

        report_history = (
            st.session_state.state.get(
                "report_version_history",
                [],
            )
        )

    if report_history:

        for item in reversed(
            report_history
        ):

            label = safe_text(
                item.get(
                    "description",
                    item.get(
                        "query",
                        "Unknown",
                    ),
                )
            )

            ts = safe_text(
                item.get(
                    "timestamp",
                    "",
                )
            )

            st.markdown(
                f'<div class="history-item">'
                f"<h4>{label[:40]}…</h4>"
                f"<span>{ts}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    else:

        st.caption(
            "No history yet."
        )

# ── Main layout ──────────────────────────────────────────────────────────────

st.title(
    "AI Research Workspace"
)

st.caption(
    "Multi-source analysis · "
    "Professional reports · "
    "Iterative refinement"
)

st.divider()

# ── Research input ───────────────────────────────────────────────────────────

col_input, col_btn = (
    st.columns([5, 1])
)

with col_input:

    query = st.text_input(
        "Research Topic",
        placeholder=(
            "e.g. Quantum Computing "
            "Applications in "
            "Drug Discovery"
        ),
        label_visibility=(
            "collapsed"
        ),
    )

with col_btn:

    generate_clicked = (
        st.button(
            "Generate →",
            use_container_width=True,
        )
    )

if generate_clicked:

    if not safe_text(query).strip():

        st.warning(
            "Please enter a research topic."
        )

    else:

        try:

            if (
                st.session_state.state
                is None
            ):

                state = (
                    StateFactory
                    .create_initial_state(
                        query=query,
                    )
                )

            else:

                state = (
                    StateManager
                    .prepare_next_workflow(
                        previous_state=(
                            st.session_state.state
                        ),
                        query=query,
                        mode=(
                            "REPORT_GENERATION"
                        ),
                    )
                )

            result = (
                run_async_workflow(
                    state=state
                )
            )

            st.session_state.state = (
                result
            )

            st.session_state.last_action = (
                "report_generated"
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Error: {error}"
            )

# ── Active report ────────────────────────────────────────────────────────────

active_report = ""

if st.session_state.state:

    raw_report = (
        st.session_state.state.get(
            "active_report",
            "",
        )
    )

    active_report = safe_text(
        raw_report
    )

if (
    st.session_state.state
    and active_report
):

    st.divider()

    (
        tab_report,
        tab_chat,
        tab_history,
    ) = st.tabs(
        [
            "📄 Report",
            "💬 Chat",
            "🕓 Versions",
        ]
    )

    # ── REPORT TAB ───────────────────────────────────────────────────────────

    with tab_report:

        if (
            st.session_state.last_action
            == "report_refined"
        ):

            st.success(
                "✓ Report refined successfully."
            )

            st.session_state.last_action = ""

        st.markdown(
            '<p class="section-label">'
            'Generated Report'
            '</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            safe_text(
                active_report
            )
        )

        st.divider()

        # ── EXPORT ───────────────────────────────────────────────────────────

        st.markdown(
            '<p class="section-label">'
            'Export'
            '</p>',
            unsafe_allow_html=True,
        )

        col_info, col_gen = (
            st.columns([3, 1])
        )

        with col_info:

            st.caption(
                "Review the report "
                "above before exporting "
                "as PDF."
            )

        with col_gen:

            if st.button(
                "Generate PDF",
                use_container_width=True,
            ):

                try:

                    state = (
                        StateManager
                        .prepare_next_workflow(
                            previous_state=(
                                st.session_state.state
                            ),
                            query=(
                                st.session_state.state[
                                    "query"
                                ]
                            ),
                            mode=(
                                "DOCUMENT_GENERATION"
                            ),
                        )
                    )

                    result = (
                        run_async_workflow(
                            state=state
                        )
                    )

                    st.session_state.state = (
                        result
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"PDF Error: {error}"
                    )

        generated_pdf = (
            st.session_state.state.get(
                "generated_pdf"
            )
        )

        if generated_pdf:

            st.success(
                "✓ PDF ready."
            )

            st.download_button(
                label=(
                    "⬇ Download PDF"
                ),
                data=generated_pdf,
                file_name=(
                    "research_report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True,
            )

        st.divider()

        # ── REFINE ───────────────────────────────────────────────────────────

        st.markdown(
            '<p class="section-label">'
            'Refine Report'
            '</p>',
            unsafe_allow_html=True,
        )

        refinement_query = (
            st.text_area(
                "Refinement Instructions",
                value=(
                    st.session_state
                    .refinement_input
                ),
                placeholder=(
                    "e.g. Add recent "
                    "industry applications, "
                    "security challenges, "
                    "and future trends."
                ),
                height=120,
                label_visibility=(
                    "collapsed"
                ),
            )
        )

        if st.button(
            "Apply Refinement →",
            use_container_width=True,
        ):

            if not safe_text(
                refinement_query
            ).strip():

                st.warning(
                    "Please enter refinement instructions."
                )

            else:

                try:

                    state = (
                        StateManager
                        .prepare_next_workflow(
                            previous_state=(
                                st.session_state.state
                            ),
                            query=(
                                refinement_query
                            ),
                            mode=(
                                "REPORT_REFINEMENT"
                            ),
                            refinement_query=(
                                refinement_query
                            ),
                        )
                    )

                    result = (
                        run_async_workflow(
                            state=state
                        )
                    )

                    st.session_state.state = (
                        result
                    )

                    st.session_state.refinement_input = ""

                    st.session_state.last_action = (
                        "report_refined"
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Refinement Error: {error}"
                    )

    # ── CHAT TAB ─────────────────────────────────────────────────────────────

    with tab_chat:

        st.markdown(
            '<p class="section-label">'
            'Ask Questions About This Report'
            '</p>',
            unsafe_allow_html=True,
        )

        col_q, col_ask = (
            st.columns([5, 1])
        )

        with col_q:

            chat_query = (
                st.text_input(
                    "Question",
                    value=(
                        st.session_state.chat_input
                    ),
                    placeholder=(
                        "e.g. What were "
                        "the main technical findings?"
                    ),
                    label_visibility=(
                        "collapsed"
                    ),
                )
            )

        with col_ask:

            ask_clicked = (
                st.button(
                    "Ask →",
                    use_container_width=True,
                )
            )

        if ask_clicked:

            if not safe_text(
                chat_query
            ).strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                try:

                    state = (
                        StateManager
                        .prepare_next_workflow(
                            previous_state=(
                                st.session_state.state
                            ),
                            query=(
                                chat_query
                            ),
                            mode=(
                                "REPORT_CHAT"
                            ),
                            report_chat_query=(
                                chat_query
                            ),
                        )
                    )

                    result = (
                        run_async_workflow(
                            state=state
                        )
                    )

                    st.session_state.state = (
                        result
                    )

                    st.session_state.chat_input = ""

                    st.rerun()

                except Exception as error:

                    st.error(
                        f"Chat Error: {error}"
                    )

        report_chat_response = ""

        if st.session_state.state:

            raw_response = (
                st.session_state.state.get(
                    "report_chat_response",
                    "",
                )
            )

            report_chat_response = (
                safe_text(
                    raw_response
                )
            )

        if report_chat_response:

            st.divider()

            st.markdown(
                '<p class="section-label">'
                'Answer'
                '</p>',
                unsafe_allow_html=True,
            )

            st.info(
                safe_text(
                    report_chat_response
                )
            )

    # ── VERSION HISTORY ──────────────────────────────────────────────────────

    with tab_history:

        version_history = (
            st.session_state.state.get(
                "report_version_history",
                [],
            )
        )

        if not version_history:

            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">◎</div>
                    <p>
                        No versions yet —
                        generate or refine a report
                        to begin.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<p class="section-label">'
                'Version History'
                '</p>',
                unsafe_allow_html=True,
            )

            st.caption(
                f"{len(version_history)} "
                f"version(s) saved"
            )

            for idx, version in enumerate(
                reversed(version_history),
                start=1,
            ):

                desc = safe_text(
                    version.get(
                        "description",
                        version.get(
                            "mode",
                            "Unknown",
                        ),
                    )
                )

                ts = safe_text(
                    version.get(
                        "timestamp",
                        "",
                    )
                )

                report_text = safe_text(
                    version.get(
                        "report",
                        "",
                    )
                )

                with st.expander(
                    f"Version {idx} · "
                    f"{desc} · {ts}"
                ):

                    if version.get(
                        "updated_section"
                    ):

                        st.caption(
                            f"Updated section: "
                            f"{version['updated_section']}"
                        )

                        st.divider()

                    st.markdown(
                        safe_text(
                            report_text
                        )
                    )

                    st.divider()

                    if st.button(
                        (
                            "Generate & Download PDF "
                            f"— Version {idx}"
                        ),
                        key=(
                            f"download_version_{idx}"
                        ),
                        use_container_width=True,
                    ):

                        try:

                            temp_state = dict(
                                st.session_state.state
                            )

                            temp_state[
                                "active_report"
                            ] = report_text

                            document_state = (
                                StateManager
                                .prepare_next_workflow(
                                    previous_state=(
                                        temp_state
                                    ),
                                    query=(
                                        temp_state[
                                            "query"
                                        ]
                                    ),
                                    mode=(
                                        "DOCUMENT_GENERATION"
                                    ),
                                )
                            )

                            result = (
                                run_async_workflow(
                                    state=(
                                        document_state
                                    )
                                )
                            )

                            generated_pdf = (
                                result.get(
                                    "generated_pdf"
                                )
                            )

                            if generated_pdf:

                                st.success(
                                    "✓ PDF generated successfully."
                                )

                                safe_name = (
                                    desc.lower()
                                    .replace(
                                        " ",
                                        "_",
                                    )
                                    .replace(
                                        "/",
                                        "_",
                                    )
                                )

                                st.download_button(
                                    label=(
                                        "⬇ Download PDF"
                                    ),
                                    data=(
                                        generated_pdf
                                    ),
                                    file_name=(
                                        f"{safe_name}_version_{idx}.pdf"
                                    ),
                                    mime=(
                                        "application/pdf"
                                    ),
                                    key=(
                                        f"pdf_download_{idx}"
                                    ),
                                    use_container_width=True,
                                )

                        except Exception as error:

                            st.error(
                                f"Version PDF Error: {error}"
                            )

else:

    if not st.session_state.state:

        st.markdown(
            """
            <div class="empty-state"
                 style="margin-top:4rem;">
                <div class="empty-state-icon">
                    ⬡
                </div>
                <p>
                    Enter a topic above and
                    press Generate to begin
                    your research.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )