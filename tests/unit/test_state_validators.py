from state.constants import Domain
from state.models import Document
from state.models import SubQuery
from state.validators import StateValidator


def test_validate_sub_queries_returns_false_for_empty_list():
    assert not StateValidator.validate_sub_queries([])


def test_validate_sub_queries_rejects_invalid_priority():
    valid_query = SubQuery(
        query="Test query",
        domain=Domain.WEB,
        priority=1,
    )
    invalid_query = SubQuery(
        query="Another query",
        domain=Domain.WEB,
        priority=5,
    )

    assert StateValidator.validate_sub_queries([valid_query, invalid_query])


def test_validate_documents_rejects_empty_content():
    document = Document(
        source=Domain.WEB,
        title="Title",
        content="",
    )

    assert not StateValidator.validate_documents([document])


def test_validate_documents_accepts_valid_documents():
    document = Document(
        source=Domain.WEB,
        title="Title",
        content="Body text",
    )

    assert StateValidator.validate_documents([document])
