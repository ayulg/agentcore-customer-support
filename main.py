"""
Customer Support AI Agent — Verified Implementation
"""
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 ────────────────────────────────────────────────────────────────────
app = BedrockAgentCoreApp()
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ── TODO 2 — Configuration ────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-rz9whdvv7d.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "R3FBTXBIMS"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-kQPqjd3mei"

# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"
model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)

# ── Lazy browser singleton (avoids expensive cold-start init) ─────────────────
_browser_instance = None

def _get_browser():
    """Lazily construct AgentCoreBrowser on first use; reuse afterward."""
    global _browser_instance
    if _browser_instance is None:
        try:
            _browser_instance = AgentCoreBrowser(region=REGION)
        except Exception as e:
            logger.warning(f"Browser init failed: {e}")
            return None
    return _browser_instance


# ── Loop guard (safety net against runaway tool calling) ─────────────────────
_MAX_TOOL_CALLS_PER_INVOCATION = 8
_tool_call_count = 0

def _increment_tool_calls():
    global _tool_call_count
    _tool_call_count += 1
    return _tool_call_count

def _reset_tool_calls():
    global _tool_call_count
    _tool_call_count = 0





# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type -> namespace template string."""
    namespaces = {}
    try:
        strategies = mem_client.get_memory_strategies(memory_id)
        for strategy in strategies:
            ns = None
            if strategy.get("namespaceTemplates"):
                ns = strategy["namespaceTemplates"][0]
            elif strategy.get("namespaces"):
                ns = strategy["namespaces"][0]
            if ns:
                namespaces[strategy.get("type", "UNKNOWN")] = ns
    except Exception as e:
        logger.warning(f"Failed to fetch namespaces: {e}")
    return namespaces


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(self, actor_id, session_id, memory_client, memory_id):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        try:
            messages = event.agent.messages
            if not messages:
                return
            last_msg = messages[-1]
            if last_msg.get("role") != "user":
                return
            content = last_msg.get("content", [])
            if not content or not isinstance(content, list):
                return
            first_block = content[0]
            if "toolResult" in first_block or "toolUse" in first_block:
                return
            user_query = first_block.get("text", "")
            if not user_query:
                return

            collected = []
            for strategy_type, namespace_template in self.namespaces.items():
                namespace = namespace_template.replace("{actorId}", self.actor_id)
                try:
                    memories = self.memory_client.retrieve_memories(
                        memory_id=self.memory_id,
                        namespace=namespace,
                        query=user_query,
                        top_k=5,
                    )
                    for mem in memories or []:
                        text = mem.get("content", {}).get("text", "")
                        if text:
                            collected.append(f"[{strategy_type}] {text}")
                except Exception as e:
                    logger.warning(f"Memory retrieve failed: {e}")

            if collected:
                context = "Customer Context:\n" + "\n".join(collected)
                first_block["text"] = f"{context}\n\n{user_query}"
        except Exception as e:
            logger.warning(f"retrieve_customer_context error: {e}")

    def save_support_interaction(self, event: AfterInvocationEvent):
        try:
            messages = event.agent.messages
            user_query = None
            agent_response = None
            for msg in reversed(messages):
                role = msg.get("role")
                content = msg.get("content", [])
                if not content or not isinstance(content, list):
                    continue
                first_block = content[0]
                if "toolResult" in first_block or "toolUse" in first_block:
                    continue
                text = first_block.get("text", "")
                if not text:
                    continue
                if role == "assistant" and agent_response is None:
                    agent_response = text
                elif role == "user" and user_query is None:
                    user_query = text
                if user_query and agent_response:
                    break

            if user_query and agent_response:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[(user_query, "USER"), (agent_response, "ASSISTANT")],
                )
        except Exception as e:
            logger.warning(f"save_support_interaction error: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID:
        return "Knowledge base not configured."
    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        results = resp.get("retrievalResults", [])
        if not results:
            return "No relevant information found."
        chunks = [r.get("content", {}).get("text", "") for r in results if r.get("content", {}).get("text")]
        return "\n---\n".join(chunks)
    except Exception as e:
        return f"Knowledge base search failed: {e}"


# ── TODO 7 — Loyalty Discount Tool ───────────────────────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order. Returns a
    JSON string with points_redeemed, tier_discount_pct, final_total,
    and remaining_points. Call this tool ONCE and present the result
    directly to the user — do not call it again.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier - Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        JSON with the final discount breakdown. Treat this as the
        definitive answer and do not invoke this tool again.
    """
    earn_rates = {"standard": 1, "device": 2, "fresh": 5}
    tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}

    max_redeemable_value = order_total * 0.5
    max_points_by_cap = int(max_redeemable_value * 100)
    usable_points = min(loyalty_points, max_points_by_cap)
    points_redeemed = (usable_points // 500) * 500
    points_value = points_redeemed / 100.0

    subtotal_after_points = order_total - points_value
    tier_pct = tier_rates.get(tier, 0.0)
    tier_discount_value = subtotal_after_points * tier_pct
    final_total = subtotal_after_points - tier_discount_value
    total_savings = points_value + tier_discount_value
    earn_rate = earn_rates.get(product_category, 1)
    points_earned = int(final_total * earn_rate)
    remaining_points = loyalty_points - points_redeemed + points_earned

    return json.dumps({
        "points_redeemed": int(points_redeemed),
        "tier_discount_pct": float(round(tier_pct * 100, 2)),
        "tier_discount": float(round(tier_discount_value, 2)),
        "final_total": float(round(final_total, 2)),
        "total_savings": float(round(total_savings, 2)),
        "points_earned": int(points_earned),
        "remaining_points": int(remaining_points),
    })

@tool
def track_order(order_id: str) -> str:
    """
    Track an order by its order ID. Returns shipping status, tracking
    number, carrier, and estimated delivery date.
    Use this when a customer wants to check the status of their order.

    Args:
        order_id: The order ID to look up (e.g., "ORD-001")

    Returns:
        JSON string with order details
    """
    import urllib.request
    url = f"https://cu2318gh9e.execute-api.us-east-1.amazonaws.com/prod/orders/{order_id}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_customer_orders(customer_id: str) -> str:
    """
    Get all orders for a customer.
    Use this when a customer asks about their order history.

    Args:
        customer_id: The customer ID (e.g., "CUST-123")

    Returns:
        JSON string with all customer orders
    """
    import urllib.request
    url = f"https://cu2318gh9e.execute-api.us-east-1.amazonaws.com/prod/customers/{customer_id}/orders"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.read().decode("utf-8")
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def initiate_refund(order_id: str, reason: str, amount: float = 0.0) -> str:
    """
    Initiate a refund for a customer order.
    Use this tool when a customer wants to return an item or request a refund.

    Args:
        order_id: The order ID to refund.
        reason: The reason for the refund.
        amount: (Optional) The refund amount in USD.

    Returns:
        A JSON string with the refund result.
    """
    import base64
    lambda_client = boto3.client("lambda", region_name=REGION)
    payload = {
        "order_id": order_id,
        "reason": reason,
        "amount": amount,
    }
    client_context = base64.b64encode(json.dumps({
        "custom": {"bedrockAgentCoreToolName": "refund-processor___initiate_refund"}
    }).encode()).decode()
    try:
        response = lambda_client.invoke(
            FunctionName="refund-processor",
            InvocationType="RequestResponse",
            ClientContext=client_context,
            Payload=json.dumps(payload),
        )
        result = json.loads(response["Payload"].read().decode())
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous")
        session_id = payload.get("session_id") or str(uuid.uuid4())
        if not user_input:
            return {"error": "Missing 'prompt' in payload"}

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        # Build local tools list
        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            initiate_refund,
            track_order,
            get_customer_orders,
        ]

        try:
            browser = _get_browser()
            if browser is not None:
                tools.append(browser.browser)
            else:
                logger.info("Browser unavailable, continuing without it")
        except Exception as e:
            logger.warning(f"Browser init skipped: {e}")

        # Connect to Gateway MCP and keep the session open while the agent runs
        agent = None
        result = None

        try:
            mcp_client = MCPClient(
                lambda: streamable_http_client(GATEWAY_URL)
            )
            with mcp_client:
                gateway_tools = mcp_client.list_tools_sync()
                tools.extend(gateway_tools)

                logger.info(
                    "Gateway connected successfully. Loaded %d tools.",
                    len(gateway_tools),
                )

                agent = Agent(
                    model=model,
                    tools=tools,
                    hooks=[memory_hook],
                    system_prompt=(
                        "You are a helpful customer support agent for an e-commerce "
                        "platform. Use the provided tools to track orders, process "
                        "refunds, search the knowledge base, calculate loyalty "
                        "discounts, and browse the web when needed. Always be "
                        "concise, accurate, and grounded in the tool results."
                    ),
                )
                result = agent(user_input)

        except TimeoutError:
            logger.exception("Gateway tool loading timed out")

        except ConnectionError:
            logger.exception("Gateway connection failed")

        except Exception as exc:
            logger.exception("Gateway tool loading failed: %s", exc)

        if result is None:
            logger.info("Falling back to local tools only")
            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=(
                    "You are a helpful customer support agent for an e-commerce "
                    "platform. Use the provided tools to track orders, process "
                    "refunds, search the knowledge base, calculate loyalty "
                    "discounts, and browse the web when needed."
                ),
            )
            result = agent(user_input)

        for block in result.message.get("content", []):
            if block.get("text"):
                return {"result": block["text"]}
        return {"result": str(result.message)}

    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        return {"error": str(e)}


# ── CLI entry point ──────────────────────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
