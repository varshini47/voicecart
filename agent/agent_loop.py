"""The agent's decision loop: reply, call a tool, or ask a clarifying question.

Replaces the Week 1 echo agent (see CLAUDE.md's agent behavior requirements).
Each turn can involve multiple rounds of tool calls before the LLM produces
a final spoken reply; MAX_TOOL_ROUNDS bounds that so a confused model can't
loop forever within a single request.
"""

from __future__ import annotations

import json

from agent import llm, session
from agent.mcp_client import MCPClient

MAX_TOOL_ROUNDS = 6

FALLBACK_REPLY = "Sorry, I'm having trouble completing that. Could you try again?"

SYSTEM_PROMPT = (
    "You are VoiceCart, a voice-based grocery shopping assistant. Keep replies "
    "short and conversational, since they will be spoken aloud.\n\n"
    "You have tools to search the catalog and manage the user's cart. Always "
    "use search_products to find a product's product_id before adding it to "
    "the cart — never guess a product_id.\n\n"
    "If a search returns multiple different brands or variants matching what "
    "the user asked for and they didn't specify one, ask a short clarifying "
    "question (e.g. \"Amul or Nandini milk?\") instead of picking one yourself. "
    "Do not call add_to_cart until that's resolved. Do not resolve it "
    "yourself by re-searching with a narrower guess — ask instead. If the "
    "user then answers your question, that resolves the same request; do "
    "not also keep an earlier item you added for that request.\n\n"
    "Quantities are per the product's listed unit (e.g. \"Eggs 6pc Tray\" is "
    "1 unit = 6 eggs, so half a dozen eggs is quantity 1, not 6). Convert "
    "casual phrases (a dozen, half a dozen, a couple) to the correct integer "
    "quantity of that unit. If genuinely unclear, ask rather than guess.\n\n"
    "To remove an item, first search_products for it (unless you already "
    "know its product_id), then check get_cart for a matching product_id. "
    "If it is in the cart, call remove_from_cart. If not, tell the user it "
    "isn't in their cart — don't call remove_from_cart for something "
    "never added.\n\n"
    "Never call checkout with confirm=true unless the user has explicitly "
    "confirmed placing the order in this conversation — even on the first "
    "message, even if the cart is empty. \"Check out my order\" alone is "
    "not confirmation. Always get_cart first, read back the cart and total "
    "(or say it's empty), and wait for an explicit yes. get_cart and "
    "search_products are read-only and never need confirmation.\n\n"
    "If a tool call returns an error, you may retry once with corrected "
    "arguments if you can identify the fix. If it fails again, or you can't "
    "identify a fix, explain the problem to the user in plain, non-technical "
    "language rather than trying further."
)


async def run_turn(session_id: str, user_text: str, mcp_client: MCPClient) -> str:
    history = session.get_history(session_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_text}]

    turn_messages = [{"role": "user", "content": user_text}]
    reply_text = FALLBACK_REPLY

    for _ in range(MAX_TOOL_ROUNDS):
        assistant_message = llm.chat_completion(messages, tools=mcp_client.tool_schemas)
        messages.append(assistant_message)
        turn_messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls")
        if not tool_calls:
            reply_text = assistant_message["content"]
            break

        for call in tool_calls:
            name = call["function"]["name"]
            arguments = json.loads(call["function"]["arguments"]) or {}
            if name in mcp_client.tools_requiring_session:
                arguments["session"] = session_id
            result = await mcp_client.call_tool(name, arguments)
            tool_message = {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result),
            }
            messages.append(tool_message)
            turn_messages.append(tool_message)

    for message in turn_messages:
        session.append(session_id, message)

    return reply_text
