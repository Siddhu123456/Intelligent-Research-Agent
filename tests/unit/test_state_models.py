import pytest

from state.constants import Domain
from state.models import Citation
from state.models import Document
from state.models import SubQuery


def test_subquery_accepts_string_domain():
    subquery = SubQuery(
        query="Quantum computing",
        domain="web",
        priority=1,
    )

    assert subquery.domain == Domain.WEB
    assert subquery.priority == 1
    assert subquery.query == "Quantum computing"


def test_subquery_invalid_query_length_raises():
    with pytest.raises(Exception):
        SubQuery(
            query="hi",
            domain="web",
            priority=1,
        )


def test_document_accepts_domain_string():
    document = Document(
        source="wikipedia",
        title="Test article",
        content="Document content",
    )

    assert document.source == Domain.WIKIPEDIA
    assert document.title == "Test article"
    assert document.content == "Document content"


def test_citation_model_fields():
    citation = Citation(
        doc_id="doc-1",
        claim="A claim",
        source="wikipedia",
    )

    assert citation.doc_id == "doc-1"
    assert citation.claim == "A claim"
    assert citation.source == "wikipedia"
