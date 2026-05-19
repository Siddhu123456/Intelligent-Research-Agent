from enum import Enum


class Domain(str, Enum):
    WEB = "web"
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"
    VECTOR_DB = "vector_db"


class CurrentStep(str, Enum):
    START = "start"
    CONTEXTUALIZED = "contextualized"
    DECOMPOSED = "decomposed"
    RETRIEVED = "retrieved"
    ANALYZED = "analyzed"
    DONE = "done"
    ERROR = "error"