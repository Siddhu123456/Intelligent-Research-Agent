from unittest.mock import (
    ANY,
    patch,
)

from utils.llm_factory import (
    LLMFactory,
)


class TestLLMFactory:
    """Unit tests for LLMFactory."""

    @patch(
        "utils.llm_factory.ChatGroq"
    )
    def test_create_qwen_llm_default_temperature(
        self,
        mock_chat_groq,
    ):
        """Test Qwen LLM creation with default temperature."""

        mock_instance = (
            mock_chat_groq.return_value
        )

        result = (
            LLMFactory
            .create_qwen_llm()
        )

        mock_chat_groq.assert_called_once_with(
            model="qwen/qwen3-32b",
            temperature=0.0,
            api_key=ANY,
        )

        assert (
            result
            == mock_instance
        )

    @patch(
        "utils.llm_factory.ChatGroq"
    )
    def test_create_qwen_llm_custom_temperature(
        self,
        mock_chat_groq,
    ):
        """Test Qwen LLM with custom temperature."""

        LLMFactory.create_qwen_llm(
            temperature=0.7
        )

        _, kwargs = (
            mock_chat_groq.call_args
        )

        assert (
            kwargs["temperature"]
            == 0.7
        )

        assert (
            kwargs["model"]
            == "qwen/qwen3-32b"
        )

    @patch(
        "utils.llm_factory.ChatOllama"
    )
    def test_create_ollama_llm_default_temperature(
        self,
        mock_chat_ollama,
    ):
        """Test Ollama LLM creation with default temperature."""

        mock_instance = (
            mock_chat_ollama.return_value
        )

        result = (
            LLMFactory
            .create_ollama_llm()
        )

        mock_chat_ollama.assert_called_once_with(
            model="gemma:2b",
            temperature=0.2,
        )

        assert (
            result
            == mock_instance
        )

    @patch(
        "utils.llm_factory.ChatOllama"
    )
    def test_create_ollama_llm_custom_temperature(
        self,
        mock_chat_ollama,
    ):
        """Test Ollama LLM with custom temperature."""

        LLMFactory.create_ollama_llm(
            temperature=0.8
        )

        _, kwargs = (
            mock_chat_ollama.call_args
        )

        assert (
            kwargs["temperature"]
            == 0.8
        )

        assert (
            kwargs["model"]
            == "gemma:2b"
        )

    @patch(
        "utils.llm_factory.ChatGroq"
    )
    def test_create_qwen_llm_returns_chatgroq_instance(
        self,
        mock_chat_groq,
    ):
        """Test Qwen returns ChatGroq instance."""

        result = (
            LLMFactory
            .create_qwen_llm()
        )

        assert (
            result
            == mock_chat_groq.return_value
        )

    @patch(
        "utils.llm_factory.ChatOllama"
    )
    def test_create_ollama_llm_returns_chatollama_instance(
        self,
        mock_chat_ollama,
    ):
        """Test Ollama returns ChatOllama instance."""

        result = (
            LLMFactory
            .create_ollama_llm()
        )

        assert (
            result
            == (
                mock_chat_ollama
                .return_value
            )
        )