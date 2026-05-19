from uuid import uuid4

from pydantic import BaseModel, field_validator
from pydantic import ConfigDict
from pydantic import Field

from state.constants import Domain


class SubQuery(BaseModel):
    """Represents a decomposed research sub-query."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid4()),
    )

    query: str = Field(
        min_length=3,
        max_length=500,
    )

    domain: Domain

    priority: int = Field(
        ge=1,
        le=5,
    )

    @field_validator(
        "domain",
        mode="before",
    )
    @classmethod
    def validate_domain(
        cls,
        value: str | Domain,
    ) -> Domain:
        if isinstance(
            value,
            Domain,
        ):
            return value

        return Domain(value)
    
    
class SubQueryList(BaseModel):
    """Structured response model for decomposition."""

    sub_queries: list[SubQuery]


class Document(BaseModel):
    """Represents a retrieved research document."""

    model_config = ConfigDict(
        strict=True,
    )

    id: str = Field(
        default_factory=lambda: str(uuid4()),
    )
    
    source: Domain

    @field_validator(
        "source",
        mode="before",
    )
    @classmethod
    def validate_source(
        cls,
        value: str | Domain,
    ) -> Domain:
        if isinstance(
            value,
            Domain,
        ):
            return value

        return Domain(value)

    title: str = Field(
        min_length=1,
    )

    content: str = Field(
        min_length=1,
        max_length=4000,
    )

    url: str | None = None

    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, str] = Field(
        default_factory=dict,
    )


class Citation(BaseModel):
    """Represents citation mapping for findings."""

    model_config = ConfigDict(
        strict=True,
    )

    doc_id: str

    claim: str = Field(
        min_length=3,
    )

    source: str = Field(
        min_length=3,
    )
    

class RewrittenQuery(BaseModel):
    """Structured output for contextual query rewriting."""

    rewritten_query: str = Field(
        min_length=3,
        description="Standalone contextualized research query.",
    )
