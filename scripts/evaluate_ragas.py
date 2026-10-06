# scripts/evaluate_ragas.py

import asyncio
import json
from pathlib import Path

from ragas import EvaluationDataset, aevaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.collections import Faithfulness

from app.agent.agent import create_biomedical_agent
from app.agent.agent import extract_message_content
from app.rag.schemas import RAGResponse


# ================================================================
# CONFIG
# ================================================================

DATASET_PATH = Path(
    "data/evaluation/ragas_dataset.json"
)


# ================================================================
# LOAD DATASET
# ================================================================

def load_test_cases():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ================================================================
# RUN AGENT
# ================================================================

async def run_agent_question(
    agent,
    question: str,
):

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    messages = result.get(
        "messages",
        [],
    )

    # ------------------------------------------------------------
    # Extract retrieved contexts
    # ------------------------------------------------------------

    retrieved_contexts = []

    for message in messages:

        if type(message).__name__ != "ToolMessage":
            continue

        content = getattr(
            message,
            "content",
            "",
        )

        if isinstance(
            content,
            str,
        ):

            try:

                data = json.loads(content)

            except json.JSONDecodeError:

                continue

            chunks = data.get(
                "chunks",
                [],
            )

            for chunk in chunks:

                text = chunk.get(
                    "source_text",
                    "",
                )

                if text:

                    retrieved_contexts.append(
                        text
                    )

    # ------------------------------------------------------------
    # Get final response
    # ------------------------------------------------------------

    final_message = messages[-1]

    final_text = extract_message_content(
        final_message
    )

    try:

        rag_response = RAGResponse.model_validate_json(
            final_text
        )

        answer = rag_response.answer

    except Exception:

        answer = final_text

    return (
        answer,
        retrieved_contexts,
    )


# ================================================================
# MAIN
# ================================================================

async def main():

    print("=" * 70)
    print("RAGAS EVALUATION")
    print("=" * 70)

    test_cases = load_test_cases()

    print()
    print(
        f"Test cases: {len(test_cases)}"
    )

    # ------------------------------------------------------------
    # Create agent
    # ------------------------------------------------------------

    print()
    print("Starting biomedical agent...")

    agent, client = (
        await create_biomedical_agent()
    )

    samples = []

    try:

        for index, test_case in enumerate(
            test_cases,
            start=1,
        ):

            question = test_case[
                "question"
            ]

            reference = test_case[
                "reference"
            ]

            print()
            print("-" * 70)
            print(
                f"TEST CASE {index}"
            )
            print("-" * 70)

            print(
                f"Question: {question}"
            )

            answer, contexts = (
                await run_agent_question(
                    agent,
                    question,
                )
            )

            print(
                f"Retrieved contexts: "
                f"{len(contexts)}"
            )

            print(
                f"Answer: {answer}"
            )

            # ----------------------------------------------------
            # Ragas sample
            # ----------------------------------------------------

            samples.append(
                SingleTurnSample(
                    user_input=question,
                    response=answer,
                    reference=reference,
                    retrieved_contexts=contexts,
                )
            )

        # --------------------------------------------------------
        # Dataset
        # --------------------------------------------------------

        dataset = EvaluationDataset(
            samples=samples
        )

        # --------------------------------------------------------
        # Metric
        # --------------------------------------------------------

        metrics = [
            Faithfulness()
        ]

        # --------------------------------------------------------
        # Evaluate
        # --------------------------------------------------------

        print()
        print("=" * 70)
        print("RUNNING RAGAS")
        print("=" * 70)

        result = await aevaluate(
            dataset=dataset,
            metrics=metrics,
            show_progress=True,
        )

        print()
        print("=" * 70)
        print("RAGAS RESULTS")
        print("=" * 70)

        print(
            result
        )

        print()
        print(
            result.to_pandas()
        )

    finally:

        close_method = getattr(
            client,
            "close",
            None,
        )

        if close_method:

            close_result = close_method()

            if asyncio.iscoroutine(
                close_result
            ):

                await close_result


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )