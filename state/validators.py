from state.models import Document
from state.models import SubQuery


class StateValidator:
    """Provides validation utilities for state objects."""

    @staticmethod
    def validate_sub_queries(
        sub_queries: list[SubQuery],
    ) -> bool:
        if not sub_queries:
            return False

        return all(
            query.priority >= 1
            for query in sub_queries
        )

    @staticmethod
    def validate_documents(
        documents: list[Document],
    ) -> bool:
        return all(
            document.content
            for document in documents
        )