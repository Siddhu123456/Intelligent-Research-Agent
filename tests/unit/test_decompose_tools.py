from state.models import SubQuery
from tools.decompose_tools import (
    DecompositionTools,
)


def test_validate_sub_queries() -> None:
    queries = [
        SubQuery(
            query="Quantum computing",
            domain="web",
            priority=1,
        ),
    ]

    assert (
        DecompositionTools.validate_sub_queries(
            queries,
        )
        is True
    )


def test_classify_domain() -> None:
    domain = (
        DecompositionTools.classify_domain(
            "Research paper on AI",
        )
    )

    assert domain == "arxiv"