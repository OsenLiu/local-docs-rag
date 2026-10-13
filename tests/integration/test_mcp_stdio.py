import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def payload_from_result(result: object) -> dict[str, object]:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    text = next(item.text for item in result.content if item.type == "text")
    return json.loads(text)


def test_stdio_server_exposes_search_and_cited_answer(tmp_path: Path) -> None:
    server_script = Path(__file__).with_name("mcp_test_server.py")
    environment = os.environ | {"RAG_TEST_DATABASE_PATH": str(tmp_path / "mcp.sqlite3")}
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        env=environment,
    )

    async def call_tools() -> tuple[dict[str, object], dict[str, object]]:
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                search_result = await session.call_tool("search_documents", {"query": "reset button"})
                answer_result = await session.call_tool("answer_documents", {"query": "How long is reset?"})
                return payload_from_result(search_result), payload_from_result(answer_result)

    search_payload, answer_payload = asyncio.run(call_tools())

    search_entry = search_payload["results"][0]
    citation = answer_payload["citations"][0]
    assert search_entry["source"] == "manual.md"
    assert search_entry["locator"] == "Reset > Recovery"
    assert answer_payload["abstained"] is False
    assert citation["source"] == "manual.md"
    assert citation["locator"] == "Reset > Recovery"