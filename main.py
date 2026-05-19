import asyncio

from graph.executor import (
    GraphExecutor,
)

from memory.memory_manager import (
    MemoryManager,
)

from utils.embedding_factory import (
    EmbeddingFactory,
)

from utils.state_factory import (
    StateFactory,
)


def bootstrap() -> None:
    """Preload heavy shared resources."""

    EmbeddingFactory.get_embeddings()


def print_result(
    result,
) -> None:

    print(
        "\n===== STATUS ====="
    )

    print(
        result["current_step"]
    )

    if result.get(
        "error",
    ):

        print(
            "\n===== ERROR ====="
        )

        print(
            result["error"]
        )

        return

    print(
        "\n===== ORIGINAL QUERY ====="
    )

    print(
        result["query"]
    )

    print(
        "\n===== CONTEXTUALIZED QUERY ====="
    )

    print(
        result[
            "contextualized_query"
        ]
    )

    print(
        "\n===== FINAL REPORT =====\n"
    )

    print(
        result["final_report"]
    )

    print(
        "\n===== MEMORY SUMMARY ====="
    )

    print(
        result[
            "conversation_summary"
        ]
    )

    print(
        "\n===== WORKFLOW ====="
    )

    print(
        f"Conversation Turn: "
        f"{result['conversation_turn']}"
    )

    print(
        f"Retry Count: "
        f"{result['retry_count']}"
    )

    print(
        f"Workflow Decision: "
        f"{result['workflow_decision']}"
    )

    print(
        "\n===== RETRIEVAL ====="
    )

    print(
        f"Retrieved Documents: "
        f"{len(result['retrieved_documents'])}"
    )


async def main() -> None:

    bootstrap()

    executor = (
        GraphExecutor()
    )

    state = (
        StateFactory
        .create_initial_state(
            query=(
                "What is quantum computing?"
            ),
        )
    )

    print(
        "\n=============================="
    )

    print(
        "FIRST QUERY"
    )

    print(
        "=============================="
    )

    result = await (
        executor.run_with_state(
            state,
        )
    )

    MemoryManager.update_memory(
        state=result,
        user_query=result["query"],
        assistant_response=(
            result[
                "analysis_summary"
            ]
        ),
    )

    print_result(
        result,
    )

    print(
        "\n=============================="
    )

    print(
        "SECOND QUERY"
    )

    print(
        "=============================="
    )

    result["query"] = (
        "What are its real-world applications?"
    )

    second_result = await (
        executor.run_with_state(
            result,
        )
    )

    MemoryManager.update_memory(
        state=second_result,
        user_query=(
            second_result["query"]
        ),
        assistant_response=(
            second_result[
                "analysis_summary"
            ]
        ),
    )

    print_result(
        second_result,
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )