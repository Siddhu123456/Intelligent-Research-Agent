from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import (
    ChatOpenAI,
)
from langchain_community.chat_models import (
    ChatOllama,
)

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

    @staticmethod
    def create_ollama_llm(
        temperature: float = 0.2,
    ):

        return ChatOllama(
            model="gemma:2b",
            temperature=temperature,
        )   
    
    