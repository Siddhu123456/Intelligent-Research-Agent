from langchain_core.messages import (
    BaseMessage,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)


class ContextTools:
    """Tools for contextual query rewriting."""

    @staticmethod
    def build_conversation_context(
        messages: list[BaseMessage],
        summary: str,
    ) -> str:
        """Build conversational context."""

        recent_messages = "\n".join(
            [
                (
                    f"{message.type}: "
                    f"{message.content}"
                )
                for message in messages
            ]
        )

        return (
            f"Conversation Summary:\n"
            f"{summary}\n\n"
            f"Recent Messages:\n"
            f"{recent_messages}"
        )

    @staticmethod
    def rewrite_query(
        query: str,
        conversation_context: str,
    ) -> str:
        """Rewrite conversational query into standalone query."""

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.1,
            )
        )

        prompt = f"""
                        You are a contextual query rewriting agent.

                        Your task:
                        Convert conversational queries into
                        standalone explicit research queries.

                        Conversation Context:
                        {conversation_context}

                        Current User Query:
                        {query}

                        Rules:
                        - preserve original meaning
                        - resolve references like:
                        it, they, them, this, that
                        - make the query standalone
                        - keep the query concise
                        - do not answer the query
                        - return only valid JSON

                        Format:
                        {{
                        "rewritten_query": "..."
                        }}

                        Examples:

                        Input:
                        "What are its applications?"

                        Output:
                        {{
                        "rewritten_query":
                        "What are the applications of quantum computing?"
                        }}

                        Input:
                        "What about recent advancements?"

                        Output:
                        {{
                        "rewritten_query":
                        "What are the recent advancements in quantum computing?"
                        }}
                        """

        response = llm.invoke(
            prompt,
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "rewritten_query": query,
                },
            )
        )

        rewritten_query = (
            parsed_response.get(
                "rewritten_query",
                query,
            )
        )

        # Defensive normalization

        if not isinstance(
            rewritten_query,
            str,
        ):

            rewritten_query = str(
                rewritten_query
            )

        return (
            rewritten_query.strip()
        )

    @staticmethod
    def refine_query(
        query: str,
    ) -> str:
        """Refine research query for retrieval."""

        llm = (
            LLMFactory
            .create_qwen_llm(
                temperature=0.2,
            )
        )

        prompt = f"""
You are a research query refinement agent.

Your task:
Refine the following research query
to improve retrieval quality.

Query:
{query}

Rules:
- preserve original meaning
- make query more specific
- improve searchability
- keep the query concise
- do not answer the query
- return only valid JSON

Format:
{{
  "refined_query": "..."
}}
"""

        response = llm.invoke(
            prompt,
        )

        parsed_response = (
            JSONParser.safe_extract(
                content=(
                    response.content
                ),
                fallback={
                    "refined_query": query,
                },
            )
        )

        refined_query = (
            parsed_response.get(
                "refined_query",
                query,
            )
        )

        # Defensive normalization

        if not isinstance(
            refined_query,
            str,
        ):

            refined_query = str(
                refined_query
            )

        return (
            refined_query.strip()
        )