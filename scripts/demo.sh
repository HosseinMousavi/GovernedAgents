#!/usr/bin/env bash
# End-to-end demo against the running compose stack (no n8n needed; n8n does the same calls).
set -euo pipefail
AGENTS=${AGENTS_URL:-http://localhost:8080}
MCP=${MCP_SERVER_URL:-http://localhost:8765/mcp}
eval "$(python scripts/tokens.py)"
j() { python -c 'import json,sys; print(json.dumps(json.load(sys.stdin), indent=2))'; }

echo "## 1. Intake spec (what n8n extracts from the chat)"
SPEC='{"spec":{"order_id":1042,"category":"damaged","description":"my steering wheel cover arrived damaged","desired_outcome":"refund","product_hint":"steering wheel cover"}}'
curl -s "$AGENTS/intake/validate" -H 'content-type: application/json' -d "$SPEC" | j

echo "## 2. Resolve under the CUSTOMER token (knowledge agent cites policy, lookup_order, refund proposed)"
RES=$(curl -s "$AGENTS/resolve" -H "authorization: Bearer $CUSTOMER_TOKEN" -H 'content-type: application/json' -d "$SPEC")
echo "$RES" | j
AMOUNT=$(echo "$RES" | python -c 'import json,sys; print(json.load(sys.stdin)["amount_cents"])')

echo "## 3. Customer tries to refund directly -> denied (missing_scope)"
curl -s "$MCP" -H "authorization: Bearer $CUSTOMER_TOKEN" -H 'x-agent-name: demo' -H 'accept: application/json' -H 'content-type: application/json' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"issue_refund\",\"arguments\":{\"order_id\":1042,\"amount_cents\":$AMOUNT,\"reason\":\"damaged\"}}}" | j

echo "## 4. Manager approves (this is what n8n's approval flow sends)"
curl -s "$MCP" -H "authorization: Bearer $MANAGER_TOKEN" -H 'x-agent-name: n8n-approval-flow' -H 'accept: application/json' -H 'content-type: application/json' \
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"issue_refund\",\"arguments\":{\"order_id\":1042,\"amount_cents\":$AMOUNT,\"reason\":\"manager approved: damaged on arrival\"}}}" | j

echo "## 5. Flip the kill switch -> every tool call is refused"
docker compose exec -T mcp touch /shared/KILL
curl -s "$MCP" -H "authorization: Bearer $MANAGER_TOKEN" -H 'x-agent-name: demo' -H 'accept: application/json' -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lookup_order","arguments":{"order_id":1042}}}' | j
docker compose exec -T mcp rm /shared/KILL

echo "## 6. Traces: http://localhost:16686  (service: store-support-agent)   Access log: docker compose logs mcp | grep tool_call"
