from state.schema import ResearchState


class SupervisorTools:
    """Tools for intelligent workflow decisions."""

    @staticmethod
    def should_retry_retrieval(
        state: ResearchState,
    ) -> bool:
        return (
            state["low_confidence"]
            and (
                state["retry_count"]
                < state["max_retries"]
            )
        )

    @staticmethod
    def should_refine_query(
        state: ResearchState,
    ) -> bool:
        retrieved_documents = (
            state["retrieved_documents"]
        )

        return (
            len(retrieved_documents)
            < 3
        )

    @staticmethod
    def increment_retry(
        state: ResearchState,
    ) -> None:
        state["retry_count"] += 1