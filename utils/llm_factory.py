from langchain_groq import ChatGroq

from config.settings import settings


class LLMFactory:
    """Factory class for creating configured LLM instances."""

    @staticmethod
    def create_qwen_llm(
        temperature: float = 0.0,
    ) -> ChatGroq:
        return ChatGroq(
            model="qwen/qwen3-32b",
            temperature=temperature,
            api_key=settings.groq_api_key,
        )