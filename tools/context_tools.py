from langchain_core.messages import (
    BaseMessage,
)

from utils.llm_factory import (
    LLMFactory,
)

from state.models import RewrittenQuery


class ContextTools:
    """Tools for contextual query rewriting."""

    @staticmethod
    def build_conversation_context(
        messages: list[BaseMessage],
        summary: str,
    ) -> str:
        recent_messages = "\n".join(
            [
                f"{message.type}: "
                f"{message.content}"
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
        
        structured_llm = llm.with_structured_output(
            RewrittenQuery, 
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
- Return only rewritten query.

Examples:
"What are its applications?"
→ "What are the applications of quantum computing?"

"What about recent advancements?"
→ "What are the recent advancements in quantum computing?"
"""

        response = structured_llm.invoke(
            prompt,
        )

        return response.rewritten_query
    
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
Refine the following research query
to improve retrieval quality.

Query:
{query}

Rules:
- Keep original meaning.
- Make query more specific.
- Improve searchability.
- Return only refined query.
"""

    response = llm.invoke(
        prompt,
    )

    return response.content.strip()