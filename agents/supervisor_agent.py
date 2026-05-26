from utils.langsmith_wrapper import traceable

from state.constants import (
    CurrentStep,
)

from state.schema import (
    ResearchState,
)

from tools.supervisor_tools import (
    SupervisorTools,
)

from utils.logger import (
    setup_logger,
)


logger = setup_logger(
    __name__,
)


class SupervisorAgent:
    """Intelligent workflow supervisor."""

    REPORT_GENERATION_MODE = (
        "REPORT_GENERATION"
    )

    REPORT_REFINEMENT_MODE = (
        "REPORT_REFINEMENT"
    )

    REPORT_CHAT_MODE = (
        "REPORT_CHAT"
    )

    DOCUMENT_GENERATION_MODE = (
        "DOCUMENT_GENERATION"
    )

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

        mode = state.get(
            "mode",
            (
                SupervisorAgent
                .REPORT_GENERATION_MODE
            ),
        )

        logger.info(
            "Current mode: %s",
            mode,
        )

        # Handle workflow errors

        if state.get("error"):

            logger.warning(
                "Workflow contains error: %s",
                state["error"],
            )

            state[
                "workflow_decision"
            ] = "terminate"

            return state

        # IMPORTANT:
        # Completion must happen BEFORE
        # mode routing to avoid loops

        if (
            current_step
            == CurrentStep.DONE.value
        ):

            state[
                "workflow_decision"
            ] = "complete"

            logger.info(
                "Workflow completed",
            )

            return state

        # Direct report refinement routing

        if (
            mode
            == (
                SupervisorAgent
                .REPORT_REFINEMENT_MODE
            )
        ):

            state[
                "workflow_decision"
            ] = (
                "report_refinement"
            )

            logger.info(
                "Routing directly to "
                "report refinement",
            )

            return state

        # Direct report chat routing

        if (
            mode
            == (
                SupervisorAgent
                .REPORT_CHAT_MODE
            )
        ):

            state[
                "workflow_decision"
            ] = "report_chat"

            logger.info(
                "Routing directly to "
                "report chat agent",
            )

            return state

        # Direct document generation routing

        if (
            mode
            == (
                SupervisorAgent
                .DOCUMENT_GENERATION_MODE
            )
        ):

            state[
                "workflow_decision"
            ] = (
                "generate_document"
            )

            logger.info(
                "Routing directly to "
                "document generation",
            )

            return state

        # Handle retrieval quality

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
                ] = (
                    "retry_retrieval"
                )

                logger.info(
                    "Supervisor decided "
                    "to retry retrieval",
                )

                return state

            state[
                "workflow_decision"
            ] = "continue"

            return state

        # Handle analysis completion

        if (
            current_step
            == CurrentStep.ANALYZED.value
        ):

            if (
                mode
                == (
                    SupervisorAgent
                    .REPORT_GENERATION_MODE
                )
            ):

                state[
                    "workflow_decision"
                ] = (
                    "generate_report"
                )

                logger.info(
                    "Routing to report "
                    "generation",
                )

                return state

        # Default continuation

        state[
            "workflow_decision"
        ] = "continue"

        return state