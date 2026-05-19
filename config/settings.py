from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


os.environ["HF_TOKEN"] = os.getenv(
    "HF_TOKEN",
    ""
)


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY")
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY")
    langchain_project: str = os.getenv(
        "LANGCHAIN_PROJECT",
        "research-assistant",
    )
    langchain_endpoint: str = os.getenv(
        "LANGCHAIN_ENDPOINT",
        "https://api.smith.langchain.com",
    )
    langchain_tracing_v2: str = os.getenv(
        "LANGCHAIN_TRACING_V2",
        "true",
    )


settings = Settings()