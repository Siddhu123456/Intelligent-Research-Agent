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

import nest_asyncio
import streamlit as st

from graph.executor import (
    GraphExecutor,
)
from utils.state_factory import (
    StateFactory,
)

nest_asyncio.apply()

st.set_page_config(
    page_title=(
        "Research Agent Chatbot"
    ),
    layout="wide",
)

st.title(
    "Research Agent Chatbot"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "state" not in st.session_state:
    st.session_state.state = None

executor = GraphExecutor()

for message in st.session_state.messages:
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

    with st.chat_message(
        "assistant",
    ):
        with st.spinner(
            "Researching...",
        ):
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
                    st.session_state.state
                )

                state["query"] = query

            result = asyncio.run(
                executor.run_with_state(
                    state,
                )
            )

            response = result[
                "final_report"
            ]

            def response_generator():
                words = response.split()

                partial = ""

                for word in words:
                    partial += word + " "

                    yield partial

            streamed_response = (
                st.write_stream(
                    response_generator()
                )
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            st.session_state.state = (
                result
            )