from langsmith import traceable

from state.constants import (
    CurrentStep,
)
from state.schema import ResearchState
from tools.supervisor_tools import (
    SupervisorTools,
)
from utils.logger import setup_logger


logger = setup_logger(__name__)


class SupervisorAgent:
    """Intelligent workflow supervisor."""

    @staticmethod
    @traceable(
        name="supervisor_agent",
    )
    def run(
        state: ResearchState,
    ) -> ResearchState:
        logger.info(
            "Supervisor evaluating workflow",
        )

        current_step = (
            state["current_step"]
        )

        if (
            current_step
            == CurrentStep.RETRIEVED.value
        ):
            should_retry = (
                SupervisorTools
                .should_retry_retrieval(
                    state,
                )
            )

            if should_retry:
                SupervisorTools.increment_retry(
                    state,
                )

                state[
                    "workflow_decision"
                ] = "retry_retrieval"

                logger.info(
                    (
                        "Supervisor decided "
                        "to retry retrieval"
                    ),
                )

                return state

        if (
            current_step
            == CurrentStep.ANALYZED.value
        ):
            state[
                "workflow_decision"
            ] = "generate_report"

            return state

        state["workflow_decision"] = (
            "continue"
        )

        return state