from langgraph.graph import END

from graph.router import GraphRouter
from state.constants import CurrentStep


def test_route_start() -> None:
    state = {
        "current_step":
            CurrentStep.START.value,
        "error": None,
        "low_confidence": False,
    }

    assert (
        GraphRouter.route(state)
        == "query_decomposition"
    )


def test_route_decomposed() -> None:
    state = {
        "current_step":
            CurrentStep.DECOMPOSED.value,
        "error": None,
        "low_confidence": False,
    }

    assert (
        GraphRouter.route(state)
        == "retrieval"
    )


def test_route_retrieved() -> None:
    state = {
        "current_step":
            CurrentStep.RETRIEVED.value,
        "error": None,
        "low_confidence": False,
    }

    assert (
        GraphRouter.route(state)
        == "analysis"
    )


def test_route_analyzed() -> None:
    state = {
        "current_step":
            CurrentStep.ANALYZED.value,
        "error": None,
        "low_confidence": False,
    }

    assert (
        GraphRouter.route(state)
        == "report_generation"
    )


def test_route_done() -> None:
    state = {
        "current_step":
            CurrentStep.DONE.value,
        "error": None,
        "low_confidence": False,
    }

    assert (
        GraphRouter.route(state)
        == END
    )


def test_route_low_confidence() -> None:
    state = {
        "current_step":
            CurrentStep.RETRIEVED.value,
        "error": None,
        "low_confidence": True,
    }

    assert (
        GraphRouter.route(state)
        == "retrieval"
    )


def test_route_error() -> None:
    state = {
        "current_step":
            CurrentStep.RETRIEVED.value,
        "error": "API failure",
        "low_confidence": False,
    }

    assert (
        GraphRouter.route(state)
        == "error_recovery"
    )