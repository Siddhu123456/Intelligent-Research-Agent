from datetime import datetime

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

        state[
            "report_history"
        ].append(
            {
                "query": query,
                "report": report,
                "timestamp": (
                    datetime.utcnow()
                    .isoformat()
                ),
            }
        )