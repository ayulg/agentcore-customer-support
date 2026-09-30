# AI Customer Support Agent — Amazon Bedrock AgentCore

An intelligent customer support assistant for an e-commerce platform, built with
Amazon Bedrock AgentCore and the Strands SDK. The agent handles order tracking,
refund processing, product Q&A (RAG), cross-session memory, loyalty discount
calculations, and live web browsing.

## Contents

- `main.py` — Complete agent implementation
- `reflection.txt` — Written reflection on design and production decisions
- `lambda/` — Lambda target handlers (order_tracker, refund_processor, schema)
- `tests/` — Text logs of all 6 verified test scenarios
- `screenshots/` — Terminal screenshots for each test + code evidence
- `product_catalog.txt` — Source data for the Bedrock Knowledge Base
- `pyproject.toml` — Python dependencies

## Test Scenarios (all verified)

1. Order Tracking — returns tracking number, carrier, ETA
2. Refund Processing — returns refund ID, APPROVED, 3-5 business days
3. Knowledge Base (RAG) — returns Platinum tier benefits from the KB
4. Cross-Session Memory — second session recalls name + preference
5. Loyalty Discount — returns points redeemed, tier %, final total, remaining
6. Browser Tool — retrieves live page title from an external URL

## Implementation Notes

- Direct Lambda invocation uses a base64-encoded ClientContext header that
  carries `bedrockAgentCoreToolName` — this mimics the AgentCore Gateway's
  routing logic and works around sandbox IAM constraints on the Gateway
  execution role trust policy.
- `calculate_loyalty_discount` is implemented as pure Python (replacing the
  Code Interpreter path) to eliminate an infinite tool-calling loop.
- `AgentCoreBrowser` is lazily initialized via a singleton to avoid cold-start
  timeouts on AgentCore Runtime.

## Credits

Built as part of the Udacity AI Engineering Nanodegree.
Starter code: https://github.com/udacity/cd14763-project-starter
