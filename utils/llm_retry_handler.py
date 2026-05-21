from utils.llm_factory import LLMFactory


class LLMRetryHandler:

    @staticmethod
    def invoke_with_fallback(
        chain,
        payload,
    ):

        try:

            return chain.invoke(
                payload
            )

        except Exception as error:

            error_message = str(
                error
            ).lower()

            if (
                "413" in error_message
                or "rate_limit" in error_message
                or "tokens per minute"
                in error_message
            ):

                # fallback to ollama

                ollama_llm = (
                    LLMFactory
                    .create_ollama_llm()
                )

                fallback_chain = (
                    chain.first
                    | ollama_llm
                )

                return fallback_chain.invoke(
                    payload
                )

            raise error