# n8n workflows

Import the three JSON files (n8n -> Workflows -> Import from file), in order, then:

1. Set the Anthropic credential on **01 / Anthropic Chat Model**.
2. In **01** and **02**, point the two *Execute Workflow* nodes at the imported 02 and 03 workflows.
3. Add these environment variables to the n8n container (docker-compose does this for you):
   - `AGENTS_URL` (default `http://agents:8080`)
   - `MCP_SERVER_URL` (default `http://mcp:8765/mcp`)
   - `CUSTOMER_TOKEN` - a `customer:read_own` token for the demo customer (`make tokens` prints one)
   - `MANAGER_TOKEN` - a `staff:read_all,manager:refund` token
   - `NOTIFY_WEBHOOK_URL` - Slack/Teams incoming webhook (optional)

## What each flow does

| Flow | Role |
|---|---|
| 01 Intake chat | Chat trigger -> LLM asks the defined questions -> extracts `IssueSpec` JSON -> `POST /intake/validate` (re-asks missing fields) -> `POST /resolve` -> branches on `requires_manager`. |
| 02 Manager approval | Wait-for-form node shows order, amount, rationale and policy citations. On approve, n8n calls MCP `issue_refund` **with the manager's own token** (JSON-RPC `tools/call`). The agents never hold refund scope. |
| 03 Ticket notification | Builds a PII-free summary (ids, amounts, citations, request id) and posts it. A regex guard throws if an email ever slips in. |

Tokens in env vars are fine for a demo. In production, n8n would exchange the manager's SSO identity for a short-lived token at approval time.
