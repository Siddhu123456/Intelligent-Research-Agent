from datetime import (
    datetime,
)

from state.schema import (
    ResearchState,
)


class ReportHistoryUtility:
    """Utility for maintaining report history."""

    @staticmethod
    def add_report(
        state: ResearchState,
        query: str,
        report: str,
    ) -> None:

        report_history = (
            state.get(
                "report_history",
                [],
            )
        )

        version = (
            len(report_history) + 1
        )

        report_entry = {
            "version": version,
            "query": query,
            "report": report,
            "timestamp": (
                datetime.utcnow()
                .isoformat()
            ),
        }

        report_history.append(
            report_entry,
        )

        state["report_history"] = (
            report_history
        )

        state[
            "report_version_history"
        ].append(
            report_entry,
        )

    @staticmethod
    def get_latest_report(
        state: ResearchState,
    ) -> dict | None:

        report_history = (
            state.get(
                "report_history",
                [],
            )
        )

        if not report_history:

            return None

        return report_history[-1]