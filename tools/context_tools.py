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
        llm = (
            LLMFactory.create_qwen_llm(
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
- Preserve original meaning.
- Resolve references like:
  it, they, them, this, that.
- Make the query standalone.
- Keep it concise.
- Do not answer the query.
Return ONLY valid JSON.

Format:
{{
  "rewritten_query": "..."
}}

Examples:

"What are its applications?"
→
{{
  "rewritten_query":
  "What are the applications of quantum computing?"
}}

"What about recent advancements?"
→
{{
  "rewritten_query":
  "What are the recent advancements in quantum computing?"
}}
"""

        response = llm.invoke(
            prompt,
        )

        try:

            parsed_response = (
                JSONParser.extract_json(
                    response.content,
                )
            )

            return (
                parsed_response[
                    "rewritten_query"
                ].strip()
            )

        except Exception:

            return query

    @staticmethod
    def refine_query(
        query: str,
    ) -> str:

        llm = (
            LLMFactory.create_qwen_llm(
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
- Keep original meaning.
- Make query more specific.
- Improve searchability.
- Keep the query concise.
- Do not answer the query.
(
Return ONLY valid JSON.

Format:
{{
  "refined_query": "..."
}}
)
"""

        response = llm.invoke(
            prompt,
        )

        try:

            parsed_response = (
                JSONParser.extract_json(
                    response.content,
                )
            )

            return (
                parsed_response[
                    "refined_query"
                ].strip()
            )

        except Exception:

            return query