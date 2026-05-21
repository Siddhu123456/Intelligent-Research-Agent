import asyncio

from graph.executor import (
    GraphExecutor,
)

from utils.state_factory import (
    StateFactory,
)


async def main():

    executor = (
        GraphExecutor()
    )

    state = (
        StateFactory
        .create_initial_state(
            query=(
                "The Future "
                "of Space "
                "Exploration"
            )
        )
    )

    async for event in (
        executor.stream_workflow(
            state
        )
    ):

        print(
            "\nEVENT:"
        )

        print(event)


if __name__ == "__main__":

    asyncio.run(
        main()
    )