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

from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from graph.executor import GraphExecutor
from utils.state_factory import StateFactory


if "messages" not in st.session_state:
    st.session_state.messages = []

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


graph_executor = (
    st.session_state.graph_executor
)


# Sidebar report history

with st.sidebar:

    st.header(
        "Research History"
    )

    if (
        st.session_state.state
        and st.session_state.state.get(
            "report_history",
        )
    ):

        report_history = (
            st.session_state.state[
                "report_history"
            ]
        )

        for index, item in enumerate(
            reversed(report_history),
            start=1,
        ):

            st.markdown(
                f"### {index}. "
                f"{item['query']}"
            )

            st.caption(
                item["timestamp"]
            )

    else:

        st.info(
            "No reports generated yet."
        )


# Render chat history

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"],
    ):

        st.markdown(
            message["content"],
        )


query = st.chat_input(
    "Ask a research question...",
)


if query:

    # Render user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message(
        "user",
    ):

        st.markdown(query)

    # Generate assistant response

    with st.chat_message(
        "assistant",
    ):

        with st.spinner(
            "Researching...",
        ):

            try:

                # Create initial state

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

                # Continue existing session

                else:

                    previous_state = (
                        st.session_state.state
                    )

                    state = (
                        StateFactory
                        .create_initial_state(
                            query=query,
                        )
                    )

                    state[
                        "conversation_turn"
                    ] = (
                        previous_state.get(
                            "conversation_turn",
                            0,
                        )
                    )

                    state[
                        "conversation_summary"
                    ] = (
                        previous_state.get(
                            "conversation_summary",
                            "",
                        )
                    )

                    state[
                        "report_history"
                    ] = (
                        previous_state.get(
                            "report_history",
                            [],
                        )
                    )

                    state[
                        "session_id"
                    ] = (
                        previous_state.get(
                            "session_id",
                        )
                    )

                # Execute workflow

                def run_agent():

                    return asyncio.run(
                        graph_executor
                        .run_with_state(
                            state,
                        )
                    )

                future = (
                    st.session_state.executor
                    .submit(
                        run_agent,
                    )
                )

                result = (
                    future.result()
                )

                # Handle workflow errors

                if result.get(
                    "error",
                ):

                    response = (
                        f"Error:\n\n"
                        f"{result['error']}"
                    )

                # Extract latest report

                else:

                    report_history = (
                        result.get(
                            "report_history",
                            [],
                        )
                    )

                    if report_history:

                        response = (
                            report_history[-1][
                                "report"
                            ]
                        )

                    else:

                        response = (
                            "No response generated."
                        )

                # Render response

                st.markdown(
                    response,
                )

                # Persist UI chat history

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

                # Persist latest workflow state

                st.session_state.state = (
                    result
                )

            except Exception as error:

                st.error(
                    f"Application Error:\n\n"
                    f"{str(error)}"
                )