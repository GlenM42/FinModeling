import asyncio

from bot.secretary import agent, config, sessions


class FakeFunctionCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = FakeFunctionCall(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ]
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


class FakeChoice:
    def __init__(self, finish_reason, message):
        self.finish_reason = finish_reason
        self.message = message


class FakeResponse:
    def __init__(self, choice):
        self.choices = [choice]


class FakeMCPClient:
    def __init__(self, result_text='{"ok": true}'):
        self.result_text = result_text
        self.calls = []

    def tools_for_llm(self):
        return []

    async def call_tool(self, name, args, *, user_id):
        self.calls.append((name, args, user_id))
        return self.result_text


def _patch_create(monkeypatch, responses):
    it = iter(responses)

    async def fake_create(*args, **kwargs):
        return next(it)

    monkeypatch.setattr(agent._client.chat.completions, "create", fake_create)


def test_handle_user_message_direct_answer(monkeypatch):
    sessions.reset_session(1)
    _patch_create(monkeypatch, [
        FakeResponse(FakeChoice("stop", FakeMessage(content="Your portfolio is up 5%."))),
    ])
    mcp = FakeMCPClient()

    reply = asyncio.run(agent.handle_user_message(1, "how's my portfolio?", mcp))

    assert reply == "Your portfolio is up 5%."
    assert mcp.calls == []
    sessions.reset_session(1)


def test_handle_user_message_calls_tool_then_answers(monkeypatch):
    sessions.reset_session(2)
    tool_call = FakeToolCall("call_1", "get_bank_status", "{}")
    _patch_create(monkeypatch, [
        FakeResponse(FakeChoice("tool_calls", FakeMessage(content=None, tool_calls=[tool_call]))),
        FakeResponse(FakeChoice("stop", FakeMessage(content="Your bank has $100."))),
    ])
    mcp = FakeMCPClient(result_text='{"rows": []}')

    reply = asyncio.run(agent.handle_user_message(2, "how much is in my bank?", mcp))

    assert reply == "Your bank has $100."
    assert len(mcp.calls) == 1
    assert mcp.calls[0][0] == "get_bank_status"
    assert mcp.calls[0][2] == 2  # the real user_id is what gets passed through
    sessions.reset_session(2)


def test_handle_user_message_tool_error_is_fed_back_not_raised(monkeypatch):
    sessions.reset_session(3)
    tool_call = FakeToolCall("call_1", "deposit", "{}")
    _patch_create(monkeypatch, [
        FakeResponse(FakeChoice("tool_calls", FakeMessage(content=None, tool_calls=[tool_call]))),
        FakeResponse(FakeChoice("stop", FakeMessage(content="Something went wrong, sorry."))),
    ])

    class BoomMCPClient(FakeMCPClient):
        async def call_tool(self, name, args, *, user_id):
            raise RuntimeError("subprocess died")

    mcp = BoomMCPClient()

    reply = asyncio.run(agent.handle_user_message(3, "deposit $10 into VOO", mcp))

    assert reply == "Something went wrong, sorry."
    sessions.reset_session(3)


def test_handle_user_message_hits_max_hops_returns_fallback(monkeypatch):
    sessions.reset_session(4)
    tool_call = FakeToolCall("call_1", "get_bank_status", "{}")
    responses = [
        FakeResponse(FakeChoice("tool_calls", FakeMessage(content=None, tool_calls=[tool_call])))
        for _ in range(config.MAX_TOOL_HOPS)
    ]
    _patch_create(monkeypatch, responses)
    mcp = FakeMCPClient()

    reply = asyncio.run(agent.handle_user_message(4, "loop forever", mcp))

    assert "trouble" in reply.lower()
    assert len(mcp.calls) == config.MAX_TOOL_HOPS
    sessions.reset_session(4)
