"""
Owns the MCP server subprocess and is the ONLY place `user_id` crosses the
boundary between the LLM and the MCP tools.

Security-critical invariant: the LLM must never see or control `user_id`.
`tools_for_llm()` strips it out of every tool's schema before the model ever
sees it, and `call_tool()` always overwrites whatever the model tried to pass
with the real, Telegram-authenticated user_id — so even a hallucinated or
spoofed value from the model is discarded.
"""

import copy
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _to_openai_tool(tool) -> dict:
    schema = copy.deepcopy(tool.inputSchema)
    schema.get("properties", {}).pop("user_id", None)
    if "required" in schema:
        schema["required"] = [name for name in schema["required"] if name != "user_id"]

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


class MCPClient:
    def __init__(self):
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._tools_cache: list[dict] = []

    async def start(self) -> "MCPClient":
        params = StdioServerParameters(command=sys.executable, args=["-m", "bot.mcp_server.server"])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

        tools_result = await self._session.list_tools()
        self._tools_cache = [_to_openai_tool(tool) for tool in tools_result.tools]
        return self

    async def close(self) -> None:
        await self._stack.aclose()

    def tools_for_llm(self) -> list[dict]:
        return self._tools_cache

    async def call_tool(self, name: str, llm_args: dict, *, user_id: int) -> str:
        args = dict(llm_args)
        args["user_id"] = user_id  # last write wins — never trust the model's value
        result = await self._session.call_tool(name, args)
        return result.content[0].text if result.content else ""
