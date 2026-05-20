from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import (
    ChatOpenAI,
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
    def create_openrouter_qwen_next(
        temperature: float = 0.2,
    ):

        return ChatOpenAI(
            model=(
                "nvidia/"
                "nemotron-3-nano-omni-30b-a3b-reasoning:free"
            ),
            temperature=temperature,
            api_key=(
                settings.openrouter_api_key
            ),
            base_url=(
                "https://openrouter.ai/api/v1"
            ),
        )      
    
    