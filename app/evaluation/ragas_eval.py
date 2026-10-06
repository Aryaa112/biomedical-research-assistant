
# app/evaluation/ragas_eval.py

"""
Ragas evaluation for the Biomedical Research Assistant.

This module:
    1. Runs the existing biomedical agent.
    2. Extracts the final answer.
    3. Extracts retrieved literature chunks.
    4. Builds a Ragas EvaluationDataset.
    5. Evaluates Faithfulness.

IMPORTANT:
    This does NOT create another RAG pipeline.
    It uses the existing biomedical agent and MCP retrieval.
"""

import json

from ragas import EvaluationDataset, SingleTurnSample, aevaluate
from ragas.metrics.collections import Faithfulness

from app.agent.agent import create_biomedical_agent
from scripts.run_agent import extract_message_content


# ======================================================================
# TEST QUESTIONS
# ======================================================================

EVALUATION_QUESTIONS = [
    "What sample size was used in the Kisumu Combined Ring Study?",
    "What were the main findings of the Kisumu Combined Ring Study?",
]


# ======================================================================
# EXTRACT RETRIEVED CONTEXT
# ======================================================================

def extract_retrieved_contexts(messages: list) -> list[str]:
    """
    Extract source_text from MCP ToolMessages.

    The MCP server returns JSON:

        {
            "retrieval_status": "SUCCESS",
            "chunks": [...]
        }
    """

    contexts = []

    for message in messages:

        if type(message).__name__ != "ToolMessage":
            continue

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:
            continue

        # --------------------------------------------------------------
        # MCP normally returns JSON text
        # --------------------------------------------------------------

        if isinstance(content, str):

            try:
                data = json.loads(content)

            except json.JSONDecodeError:
                continue

            chunks = data.get(
                "chunks",
                [],
            )

            for chunk in chunks:

                source_text = chunk.get(
                    "source_text",
                    "",
                )

                if source_text:
                    contexts.append(
                        source_text
                    )

        # --------------------------------------------------------------
        # Handle list content as well
        # --------------------------------------------------------------

        elif isinstance(content, list):

            for item in content:

                if not isinstance(item, dict):
                    continue

                text = item.get(
                    "text"
                )

                if not text:
                    continue

                try:
                    data = json.loads(text)

                except json.JSONDecodeError:
                    continue

                chunks = data.get(
                    "chunks",
                    [],
                )

                for chunk in chunks:

                    source_text = chunk.get(
                        "source_text",
                        "",
                    )

                    if source_text:
                        contexts.append(
                            source_text
                        )

    return contexts


# ======================================================================
# RUN ONE QUESTION
# ======================================================================

async def run_question(
    agent,
    question: str,
):
    """
    Run the existing biomedical agent and return:

        answer
        retrieved_contexts
    """

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

    if not messages:
        raise RuntimeError(
            "Agent returned no messages."
        )

    # --------------------------------------------------------------
    # Extract final answer
    # --------------------------------------------------------------

    last_message = messages[-1]

    answer = extract_message_content(
        last_message
    )

    # --------------------------------------------------------------
    # Extract retrieval context
    # --------------------------------------------------------------

    contexts = extract_retrieved_contexts(
        messages
    )

    return answer, contexts


# ======================================================================
# BUILD RAGAS DATASET
# ======================================================================

async def build_dataset(
    agent,
):
    """
    Run all evaluation questions and create
    a Ragas EvaluationDataset.
    """

    samples = []

    for index, question in enumerate(
        EVALUATION_QUESTIONS,
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"EVALUATION QUESTION {index}"
        )
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        answer, contexts = await run_question(
            agent,
            question,
        )

        print()
        print(
            f"Retrieved contexts: "
            f"{len(contexts)}"
        )

        print()
        print(
            "Answer:"
        )

        print(
            answer
        )

        if not contexts:

            print()
            print(
                "WARNING: No retrieved contexts found."
            )

        # ----------------------------------------------------------
        # Ragas sample
        # ----------------------------------------------------------

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )

        samples.append(
            sample
        )

    return EvaluationDataset(
        samples=samples
    )


# ======================================================================
# MAIN EVALUATION
# ======================================================================

async def main():

    print()
    print("=" * 70)
    print("BIOMEDICAL RAGAS EVALUATION")
    print("=" * 70)

    print()
    print(
        f"Evaluation questions: "
        f"{len(EVALUATION_QUESTIONS)}"
    )

    # --------------------------------------------------------------
    # Create existing biomedical agent
    # --------------------------------------------------------------

    print()
    print(
        "Starting existing biomedical agent..."
    )

    agent, client = (
        await create_biomedical_agent()
    )

    try:

        # ----------------------------------------------------------
        # Build dataset from real agent runs
        # ----------------------------------------------------------

        dataset = await build_dataset(
            agent
        )

        print()
        print("=" * 70)
        print("RAGAS DATASET READY")
        print("=" * 70)

        print(
            f"Samples: "
            f"{len(dataset.samples)}"
        )

        # ----------------------------------------------------------
        # Faithfulness metric
        # ----------------------------------------------------------

        faithfulness = Faithfulness()

        print()
        print("=" * 70)
        print("RUNNING FAITHFULNESS EVALUATION")
        print("=" * 70)

        result = await aevaluate(
            dataset=dataset,
            metrics=[
                faithfulness
            ],
        )

        # ----------------------------------------------------------
        # Results
        # ----------------------------------------------------------

        print()
        print("=" * 70)
        print("RAGAS RESULTS")
        print("=" * 70)

        print(
            result
        )

        print()

        try:

            dataframe = result.to_pandas()

            print(
                dataframe
            )

        except Exception as exc:

            print(
                "Could not convert results "
                f"to pandas: {exc}"
            )

    finally:

        # ----------------------------------------------------------
        # Close MCP client
        # ----------------------------------------------------------

        close_method = getattr(
            client,
            "close",
            None,
        )

        if close_method:

            close_result = close_method()

            if hasattr(
                close_result,
                "__await__",
            ):

                await close_result


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    import asyncio

    asyncio.run(
        main()
    )

