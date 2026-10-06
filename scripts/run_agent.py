# scripts/run_agent.py

import asyncio
import json
import re

from pydantic import ValidationError

from app.agent.agent import create_biomedical_agent

from app.guardrails.input_guardrail import (
    check_input,
)

from app.guardrails.citation_guardrail import (
    validate_citations,
)

from app.guardrails.grounding_guardrail import (
    check_grounding,
)

from app.rag.schemas import (
    RAGResponse,
)


# ======================================================================
# CONFIGURATION
# ======================================================================

MAX_HISTORY_MESSAGES = 6


# ======================================================================
# MESSAGE CONTENT
# ======================================================================

def extract_message_content(
    message,
) -> str:

    content = getattr(
        message,
        "content",
        "",
    )

    if isinstance(
        content,
        str,
    ):

        return content.strip()

    if isinstance(
        content,
        list,
    ):

        parts = []

        for item in content:

            if isinstance(
                item,
                str,
            ):

                parts.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):

                text = item.get(
                    "text"
                )

                if text:

                    parts.append(
                        str(text)
                    )

        return "\n".join(
            parts
        ).strip()

    return str(
        content
    ).strip()


# ======================================================================
# PARSE JSON
# ======================================================================

def _parse_json_text(
    text: str,
):

    if not text:

        return None

    text = text.strip()

    # --------------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------------

    try:

        return json.loads(
            text
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        pass

    # --------------------------------------------------------------
    # Markdown JSON
    # --------------------------------------------------------------

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    ).strip()

    try:

        return json.loads(
            cleaned
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        pass

    return None


# ======================================================================
# NORMALIZE RETRIEVAL RESULT
# ======================================================================

def _normalize_chunk_result(
    value,
) -> list[dict]:

    if value is None:

        return []

    # --------------------------------------------------------------
    # Retrieval wrapper
    # --------------------------------------------------------------

    if isinstance(
        value,
        dict,
    ):

        status = value.get(
            "retrieval_status"
        )

        if status == "OUT_OF_CORPUS":

            return []

        if status == "ERROR":

            return []

        if "chunks" in value:

            return _normalize_chunk_result(
                value.get(
                    "chunks"
                )
            )

        # Direct chunk
        if (
            value.get("chunk_id")
            or value.get("source_text")
        ):

            return [
                value
            ]

        return []

    # --------------------------------------------------------------
    # List
    # --------------------------------------------------------------

    if isinstance(
        value,
        list,
    ):

        chunks = []

        for item in value:

            chunks.extend(
                _normalize_chunk_result(
                    item
                )
            )

        return chunks

    return []


# ======================================================================
# EXTRACT RETRIEVED CHUNKS
# ======================================================================

def extract_retrieved_chunks(
    messages: list,
) -> list[dict]:

    retrieved_chunks = []

    if not isinstance(
        messages,
        list,
    ):

        return []

    for message in messages:

        if (
            type(message).__name__
            != "ToolMessage"
        ):

            continue

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:

            continue

        # ----------------------------------------------------------
        # STRING CONTENT
        # ----------------------------------------------------------

        if isinstance(
            content,
            str,
        ):

            parsed = _parse_json_text(
                content
            )

            if parsed is not None:

                retrieved_chunks.extend(
                    _normalize_chunk_result(
                        parsed
                    )
                )

        # ----------------------------------------------------------
        # LIST CONTENT
        # ----------------------------------------------------------

        elif isinstance(
            content,
            list,
        ):

            for item in content:

                if isinstance(
                    item,
                    dict,
                ):

                    text = item.get(
                        "text"
                    )

                    if text:

                        parsed = (
                            _parse_json_text(
                                text
                            )
                        )

                        if parsed is not None:

                            retrieved_chunks.extend(
                                _normalize_chunk_result(
                                    parsed
                                )
                            )

    return retrieved_chunks


# ======================================================================
# OUT OF CORPUS
# ======================================================================

def retrieval_is_out_of_corpus(
    messages: list,
) -> bool:

    if not isinstance(
        messages,
        list,
    ):

        return False

    for message in messages:

        if (
            type(message).__name__
            != "ToolMessage"
        ):

            continue

        content = getattr(
            message,
            "content",
            None,
        )

        if not content:

            continue

        text = ""

        if isinstance(
            content,
            str,
        ):

            text = content

        elif isinstance(
            content,
            list,
        ):

            for item in content:

                if isinstance(
                    item,
                    dict,
                ):

                    item_text = item.get(
                        "text"
                    )

                    if item_text:

                        text += (
                            "\n"
                            + str(
                                item_text
                            )
                        )

        if "OUT_OF_CORPUS" in text:

            return True

    return False


# ======================================================================
# EXTRACT JSON OBJECT FROM AI RESPONSE
# ======================================================================

def extract_json_object(
    text: str,
):

    if not text:

        return None

    text = text.strip()

    parsed = _parse_json_text(
        text
    )

    if parsed is not None:

        return parsed

    # --------------------------------------------------------------
    # Search embedded JSON
    # --------------------------------------------------------------

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start == -1
        or end == -1
        or end <= start
    ):

        return None

    candidate = text[
        start:end + 1
    ]

    try:

        return json.loads(
            candidate
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        return None


# ======================================================================
# PARSE RAG RESPONSE
# ======================================================================

def parse_rag_response(
    text: str,
):

    json_data = extract_json_object(
        text
    )

    if json_data is None:

        return None

    try:

        return RAGResponse.model_validate(
            json_data
        )

    except ValidationError:

        return None


# ======================================================================
# DEBUG
# ======================================================================

def print_debug_messages(
    messages: list,
) -> None:

    print()
    print(
        "DEBUG: AGENT MESSAGE FLOW"
    )
    print(
        "-" * 70
    )

    for i, message in enumerate(
        messages
    ):

        print()
        print(
            f"MESSAGE {i + 1}"
        )

        print(
            f"Type: "
            f"{type(message).__name__}"
        )

        content = extract_message_content(
            message
        )

        if content:

            print(
                f"Content: {content}"
            )

        tool_calls = getattr(
            message,
            "tool_calls",
            None,
        )

        if tool_calls:

            print(
                "Tool calls:"
            )

            for tool_call in tool_calls:

                print(
                    f"  - "
                    f"{tool_call.get('name')}"
                )

                print(
                    "    Arguments: "
                    f"{tool_call.get('args')}"
                )


# ======================================================================
# MAIN
# ======================================================================

async def main():

    print("=" * 70)
    print(
        "BIOMEDICAL RESEARCH ASSISTANT"
    )
    print("=" * 70)

    print()
    print(
        "Starting biomedical agent..."
    )
    print()

    # --------------------------------------------------------------
    # Create agent
    # --------------------------------------------------------------

    agent, client = (
        await create_biomedical_agent()
    )

    print()
    print("=" * 70)
    print(
        "BIOMEDICAL AGENT READY"
    )
    print("=" * 70)

    print()
    print("Commands:")
    print(
        "  exit / quit  -> exit application"
    )
    print(
        "  new / reset  -> start a new conversation"
    )
    print()

    conversation_history = []

    try:

        while True:

            # ======================================================
            # INPUT
            # ======================================================

            try:

                question = input(
                    "You: "
                ).strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):

                print(
                    "\nExiting..."
                )

                break

            if not question:

                print(
                    "Please enter a question."
                )

                continue

            command = question.lower()

            # ======================================================
            # EXIT
            # ======================================================

            if command in {
                "exit",
                "quit",
            }:

                print(
                    "Exiting..."
                )

                break

            # ======================================================
            # RESET
            # ======================================================

            if command in {
                "new",
                "reset",
            }:

                conversation_history.clear()

                print()
                print(
                    "-" * 70
                )
                print(
                    "NEW CONVERSATION STARTED"
                )
                print(
                    "-" * 70
                )
                print()

                continue

            # ======================================================
            # INPUT GUARDRAIL
            # ======================================================

            input_result = check_input(
                question
            )

            if not input_result.allowed:

                print()
                print(
                    "INPUT BLOCKED"
                )
                print(
                    input_result.reason
                )
                print()

                continue

            # ======================================================
            # ADD USER MESSAGE
            # ======================================================

            conversation_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            conversation_history = (
                conversation_history[
                    -MAX_HISTORY_MESSAGES:
                ]
            )

            # ======================================================
            # AGENT
            # ======================================================

            print()
            print(
                "-" * 70
            )
            print(
                "AGENT"
            )
            print(
                "-" * 70
            )

            try:

                result = await agent.ainvoke(
                    {
                        "messages":
                            conversation_history
                    }
                )

                # ==================================================
                # RESULT VALIDATION
                # ==================================================

                if not isinstance(
                    result,
                    dict,
                ):

                    raise RuntimeError(
                        "Agent returned an unexpected result type."
                    )

                messages = result.get(
                    "messages",
                    [],
                )

                if not isinstance(
                    messages,
                    list,
                ):

                    raise RuntimeError(
                        "Agent messages are not a list."
                    )

                if not messages:

                    raise RuntimeError(
                        "Agent returned no messages."
                    )

                # ==================================================
                # DEBUG
                # ==================================================

                print_debug_messages(
                    messages
                )

                # ==================================================
                # RETRIEVAL
                # ==================================================

                retrieved_chunks = (
                    extract_retrieved_chunks(
                        messages
                    )
                )

                print()
                print(
                    f"Retrieved chunks: "
                    f"{len(retrieved_chunks)}"
                )

                # ==================================================
                # OUT OF CORPUS
                # ==================================================

                if retrieval_is_out_of_corpus(
                    messages
                ):

                    final_answer = (
                        "I could not find this information "
                        "in the available literature."
                    )

                    print()
                    print(
                        "-" * 70
                    )
                    print(
                        "FINAL ANSWER"
                    )
                    print(
                        "-" * 70
                    )
                    print(
                        final_answer
                    )

                    conversation_history.append(
                        {
                            "role": "assistant",
                            "content": final_answer,
                        }
                    )

                    conversation_history = (
                        conversation_history[
                            -MAX_HISTORY_MESSAGES:
                        ]
                    )

                    continue

                # ==================================================
                # LAST MESSAGE
                # ==================================================

                last_message = messages[-1]

                last_content = (
                    extract_message_content(
                        last_message
                    )
                )

                # ==================================================
                # STRUCTURED RESPONSE
                # ==================================================

                structured_response = (
                    parse_rag_response(
                        last_content
                    )
                )

                # ==================================================
                # CLARIFICATION / NORMAL RESPONSE
                # ==================================================

                if structured_response is None:

                    if last_content:

                        print()
                        print(
                            "-" * 70
                        )
                        print(
                            "AGENT RESPONSE"
                        )
                        print(
                            "-" * 70
                        )
                        print(
                            last_content
                        )

                        conversation_history.append(
                            {
                                "role": "assistant",
                                "content":
                                    last_content,
                            }
                        )

                        conversation_history = (
                            conversation_history[
                                -MAX_HISTORY_MESSAGES:
                            ]
                        )

                        continue

                    raise RuntimeError(
                        "Agent returned no usable response."
                    )

                # ==================================================
                # CITATION GUARDRAIL
                # ==================================================

                citations_valid, citation_reason = (
                    validate_citations(
                        response=structured_response,
                        retrieved_chunks=
                            retrieved_chunks,
                    )
                )

                if not citations_valid:

                    print()
                    print(
                        "CITATION GUARDRAIL FAILED"
                    )
                    print(
                        citation_reason
                    )
                    print(
                        "Answer rejected."
                    )

                    conversation_history.pop()

                    continue

                # ==================================================
                # MATCH CITATIONS
                # ==================================================

                citation_keys = {

                    (
                        citation.document_id,
                        citation.page_number,
                        citation.chunk_id,
                    )

                    for citation
                    in structured_response.citations
                }

                cited_chunks = [

                    chunk

                    for chunk
                    in retrieved_chunks

                    if (
                        chunk.get(
                            "document_id"
                        ),
                        chunk.get(
                            "page_number"
                        ),
                        chunk.get(
                            "chunk_id"
                        ),
                    )
                    in citation_keys
                ]

                # ==================================================
                # GROUNDING
                # ==================================================

                grounding_status, grounding_reason = (
                    check_grounding(
                        question=question,
                        answer=
                            structured_response.answer,
                        cited_chunks=
                            cited_chunks,
                    )
                )

                print()
                print(
                    f"GROUNDING: "
                    f"{grounding_status}"
                )
                print(
                    f"Reason: "
                    f"{grounding_reason}"
                )

                if grounding_status == "UNGROUNDED":

                    print()
                    print(
                        "Answer rejected because it "
                        "was not sufficiently grounded."
                    )

                    conversation_history.pop()

                    continue

                if grounding_status == "PARTIAL":

                    print()
                    print(
                        "WARNING: Answer is only "
                        "partially grounded."
                    )

                # ==================================================
                # FINAL ANSWER
                # ==================================================

                print()
                print(
                    "-" * 70
                )
                print(
                    "FINAL ANSWER"
                )
                print(
                    "-" * 70
                )

                print(
                    structured_response.answer
                )

                # ==================================================
                # CITATIONS
                # ==================================================

                print()
                print(
                    "CITATIONS:"
                )

                for citation in (
                    structured_response.citations
                ):

                    print(
                        f"- Doc: "
                        f"{citation.document_id}"
                        f" | Page: "
                        f"{citation.page_number}"
                        f" | Chunk: "
                        f"{citation.chunk_id}"
                    )

                # ==================================================
                # CONFIDENCE
                # ==================================================

                print()
                print(
                    f"CONFIDENCE: "
                    f"{structured_response.confidence}"
                )

                # ==================================================
                # MEMORY
                # ==================================================

                conversation_history.append(
                    {
                        "role": "assistant",
                        "content":
                            structured_response.answer,
                    }
                )

                conversation_history = (
                    conversation_history[
                        -MAX_HISTORY_MESSAGES:
                    ]
                )

                print()
                print(
                    f"Memory: "
                    f"{len(conversation_history)}/"
                    f"{MAX_HISTORY_MESSAGES}"
                    " messages"
                )

            except Exception as exc:

                print()
                print(
                    "AGENT ERROR"
                )
                print(
                    "-" * 70
                )
                print(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                if (
                    conversation_history
                    and conversation_history[-1].get(
                        "role"
                    ) == "user"
                ):

                    conversation_history.pop()

            print()

    finally:

        try:

            close_method = getattr(
                client,
                "close",
                None,
            )

            if close_method:

                close_result = (
                    close_method()
                )

                if asyncio.iscoroutine(
                    close_result
                ):

                    await close_result

        except Exception as exc:

            print(
                "Warning: MCP cleanup failed: "
                f"{exc}"
            )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )