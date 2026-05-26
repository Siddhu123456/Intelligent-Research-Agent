from langchain_core.messages import (
    BaseMessage,
)

from utils.json_parser import (
    JSONParser,
)

from utils.llm_factory import (
    LLMFactory,
)

from tools.prompts.context_tools_prompts import (
    CONTEXT_TOOLS_REFINE_QUERY_PROMPT,
    CONTEXT_TOOLS_REWRITE_QUERY_PROMPT,
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

        prompt = CONTEXT_TOOLS_REWRITE_QUERY_PROMPT.format(conversation_context=conversation_context, query=query)

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

        prompt = CONTEXT_TOOLS_REFINE_QUERY_PROMPT.format(query=query)

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