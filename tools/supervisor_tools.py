from state.schema import ResearchState


class SupervisorTools:
    """Tools for intelligent workflow decisions."""

    @staticmethod
    def should_retry_retrieval(
        state: ResearchState,
    ) -> bool:
        # If this run was initiated by a user-triggered regeneration,
        # avoid automatic retrieval retries to prevent retry loops.
        run_meta = state.get("run_metadata", {}) or {}
        if run_meta.get("regen_initiated"):
            return False

        return (
            state.get("low_confidence", False)
            and (
                state.get("retry_count", 0)
                < state.get("max_retries", 0)
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