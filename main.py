from pprint import pprint

from graph.executor import GraphExecutor
from utils.embedding_factory import (
    EmbeddingFactory,
)
from memory.memory_manager import (
    MemoryManager,
)

def bootstrap() -> None:
    """Preload heavy shared resources."""

    EmbeddingFactory.get_embeddings()


def main() -> None:
    executor = GraphExecutor()

    result = executor.run(
        query="What is quantum computing?",
    )

    print("\n===== STATUS =====")
    print(result["current_step"])

    if result.get("error"):
        print("\n===== ERROR =====")
        print(result["error"])

        return

    print("\n===== FINAL REPORT =====\n")

    print(result["final_report"])
    
    MemoryManager.update_memory(
        state=result,
        user_query=result["query"],
        assistant_response=(
            result["analysis_summary"]
        ),
    )
    
    print("\n===== SESSION INFO =====")

    print(
        f"Session ID: "
        f"{result['session_id']}"
    )

    print(
        f"Conversation Turn: "
        f"{result['conversation_turn']}"
    )

    print("\n===== MEMORY SUMMARY =====")

    print(
        result[
            "conversation_summary"
        ]
    )

    print(
        "\n===== CONTEXTUALIZED QUERY ====="
    )

    print(
        result[
            "contextualized_query"
        ]
    )
    
    print("\n===== WORKFLOW =====")

    print(
        f"Retries: "
        f"{result['retry_count']}"
    )

    print(
        f"Decision: "
        f"{result['workflow_decision']}"
    )
    
    print(
        "\n===== RETRIEVAL QUALITY ====="
    )

    print(
        f"Optimized Documents: "
        f"{len(result['retrieved_documents'])}"
    )

if __name__ == "__main__":
    main()