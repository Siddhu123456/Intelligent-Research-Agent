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

from concurrent.futures import (
    ThreadPoolExecutor,
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

from utils.embedding_factory import (
    EmbeddingFactory,
)


from utils.reranker_factory import (
    RerankerFactory,
)


st.set_page_config(
    page_title="AI Research Workspace",
    layout="wide",
)

# Warmup embedding model
# during application startup
# AI MODEL WARMUP

if (
    "models_warmed_up"
    not in st.session_state
):

    with st.spinner(
        "Loading AI models...",
    ):

        # Embedding warmup

        embedding_model = (
            EmbeddingFactory
            .get_embeddings()
        )

        embedding_model.embed_query(
            "warmup"
        )

        # Reranker warmup

        reranker = (
            RerankerFactory
            .get_reranker()
        )

        reranker.predict(
            [
                (
                    "warmup query",
                    "warmup document",
                )
            ]
        )

        st.session_state[
            "models_warmed_up"
        ] = True

# Session initialization

if "state" not in st.session_state:

    st.session_state.state = None

if "executor" not in st.session_state:

    st.session_state.executor = (
        ThreadPoolExecutor(
            max_workers=1,
        )
    )

if "graph_executor" not in st.session_state:

    st.session_state.graph_executor = (
        GraphExecutor()
    )

if "refinement_input" not in st.session_state:

    st.session_state.refinement_input = ""

if "chat_input" not in st.session_state:

    st.session_state.chat_input = ""

if "last_action" not in st.session_state:

    st.session_state.last_action = ""


graph_executor = (
    st.session_state.graph_executor
)


def execute_workflow(
    state,
):

    def run_workflow():

        return asyncio.run(
            graph_executor
            .run_with_state(
                state,
            )
        )

    future = (
        st.session_state.executor
        .submit(
            run_workflow,
        )
    )

    return future.result()


# Sidebar

with st.sidebar:

    st.title(
        "Research Workspace"
    )

    st.divider()

    if st.button(
        "New Research",
        use_container_width=True,
    ):

        st.session_state.state = None

        st.session_state.refinement_input = ""

        st.session_state.chat_input = ""

        st.session_state.last_action = ""

        st.rerun()

    st.divider()

    st.subheader(
        "Report History"
    )

    if (
        st.session_state.state
        and st.session_state.state.get(
            "report_version_history",
        )
    ):

        report_history = (
            st.session_state.state[
                "report_version_history"
            ]
        )

        for index, item in enumerate(
            reversed(report_history),
            start=1,
        ):

            with st.container():

                st.markdown(
                    f"### {index}. "
                    f"{item.get("description", item.get("query", "Unknown"))}"
                    )

                st.caption(
                    item.get(
                        "timestamp",
                        "No timestamp",
                    )
                )
                
                st.divider()

    else:

        st.info(
            "No reports generated yet."
        )


# Main workspace

st.title(
    "AI Research Workspace"
)

st.markdown(
    (
        "Generate professional "
        "research reports with "
        "multi-source analysis."
    )
)

st.divider()


# Research generation

query = st.text_input(
    "Enter research topic",
    placeholder=(
        "Example: Quantum Computing "
        "Applications"
    ),
)

if st.button(
    "Generate Report",
    use_container_width=True,
):

    if not query.strip():

        st.warning(
            "Please enter a "
            "research topic."
        )

    else:

        with st.spinner(
            "Generating research report...",
        ):

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
                    execute_workflow(
                        state,
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
                    (
                        "Application Error:\n\n"
                        f"{str(error)}"
                    )
                )


# Active report workspace

if (
    st.session_state.state
    and st.session_state.state.get(
        "active_report",
    )
):

    st.divider()

    workspace_tab_1, workspace_tab_2, workspace_tab_3 = (
        st.tabs(
            [
                "Research Report",
                "Report Chat",
                "Version History",
            ]
        )
    )

    # Report tab

    with workspace_tab_1:

        st.subheader(
            "Research Report"
        )

        latest_report = (
            st.session_state.state.get(
                "active_report",
                "",
            )
        )

        st.markdown(
            latest_report,
        )

        if (
            st.session_state.last_action
            == "report_refined"
        ):

            st.success(
                "Report refined successfully."
            )

            st.session_state.last_action = ""

        st.divider()

        # Human-in-the-loop PDF generation

        st.subheader(
            "Export Document"
        )

        st.info(
            (
                "Review the generated report "
                "before exporting it as PDF."
            )
        )

        if st.button(
            "Generate PDF",
            use_container_width=True,
        ):

            with st.spinner(
                "Generating PDF document...",
            ):

                try:

                    state = (
                        StateManager
                        .prepare_next_workflow(
                            previous_state=(
                                st.session_state.state
                            ),
                            query=(
                                st.session_state
                                .state["query"]
                            ),
                            mode=(
                                "DOCUMENT_GENERATION"
                            ),
                        )
                    )

                    result = (
                        execute_workflow(
                            state,
                        )
                    )

                    st.session_state.state = (
                        result
                    )

                    st.rerun()

                except Exception as error:

                    st.error(
                        (
                            "PDF Generation Error:\n\n"
                            f"{str(error)}"
                        )
                    )

        generated_pdf = (
            st.session_state.state.get(
                "generated_pdf",
            )
        )

        if generated_pdf:

            st.success(
                "PDF generated successfully."
            )

            st.download_button(
                label="Download PDF",
                data=generated_pdf,
                file_name=(
                    "research_report.pdf"
                ),
                mime=(
                    "application/pdf"
                ),
                use_container_width=True,
            )

        st.divider()

        st.subheader(
            "Refine Report"
        )

        refinement_query = (
            st.text_area(
                "Refinement Instructions",
                value=(
                    st.session_state
                    .refinement_input
                ),
                placeholder=(
                    "Example: Add recent "
                    "industry applications, "
                    "security challenges, "
                    "and future trends."
                ),
                height=140,
            )
        )

        if st.button(
            "Apply Refinement",
            use_container_width=True,
        ):

            if not refinement_query.strip():

                st.warning(
                    "Please enter a "
                    "refinement instruction."
                )

            else:

                with st.spinner(
                    "Refining report...",
                ):

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
                            execute_workflow(
                                state,
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
                            (
                                "Refinement Error:\n\n"
                                f"{str(error)}"
                            )
                        )

    # Report chat tab

    with workspace_tab_2:

        st.subheader(
            "Ask Questions About Report"
        )

        chat_query = st.text_input(
            "Ask a question",
            value=(
                st.session_state
                .chat_input
            ),
            placeholder=(
                "Example: What were the "
                "main technical findings?"
            ),
        )

        if st.button(
            "Ask Report",
            use_container_width=True,
        ):

            if not chat_query.strip():

                st.warning(
                    "Please enter a "
                    "question."
                )

            else:

                with st.spinner(
                    "Analyzing report...",
                ):

                    try:

                        state = (
                            StateManager
                            .prepare_next_workflow(
                                previous_state=(
                                    st.session_state.state
                                ),
                                query=chat_query,
                                mode=(
                                    "REPORT_CHAT"
                                ),
                                report_chat_query=(
                                    chat_query
                                ),
                            )
                        )

                        result = (
                            execute_workflow(
                                state,
                            )
                        )

                        st.session_state.state = (
                            result
                        )

                        st.session_state.chat_input = ""

                        st.rerun()

                    except Exception as error:

                        st.error(
                            (
                                "Report Chat Error:\n\n"
                                f"{str(error)}"
                            )
                        )

        report_chat_response = (
            st.session_state.state.get(
                "report_chat_response",
                "",
            )
        )

        if report_chat_response:

            st.divider()

            st.subheader(
                "Report Answer"
            )

            st.markdown(
                report_chat_response,
            )

    # Version history tab

    with workspace_tab_3:

        version_history = (
            st.session_state.state.get(
                "report_version_history",
                [],
            )
        )

        if not version_history:

            st.info(
                "No report versions available."
            )

        else:

            st.subheader(
                "Report Versions"
            )

            st.caption(
                (
                    f"Total Versions: "
                    f"{len(version_history)}"
                )
            )

            for index, version in enumerate(
                reversed(version_history),
                start=1,
            ):

                version_description = (
                    version.get(
                        "description",
                        version.get(
                            "mode",
                            "unknown",
                        ),
                    )
                )

                with st.expander(
                    (
                        f"Version {index} "
                        f"- {version_description}"
                    )
                ):
                    if version.get(
                        "updated_section",
                    ):

                        st.caption(
                            (
                                "Updated Section: "
                                f"{version['updated_section']}"
                            )
                        )

                    st.markdown(
                        "### Report Snapshot"
                    )

                    st.markdown(
                        version["report"]
                    )