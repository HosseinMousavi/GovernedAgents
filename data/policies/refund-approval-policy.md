# Refund Approval Policy (internal)

## Who may approve
Only users holding the **manager:refund** scope may issue a refund. Support agents and staff may *propose* a refund; the proposal is routed to a manager through the approval workflow.

## Thresholds
- Up to $50: single manager approval.
- $50.01 - $500: single manager approval plus a ticket reference.
- Over $500: two manager approvals.

## Constraints
- A refund may never exceed the order total.
- One refund per order. Partial refunds close the order to further refunds.
- Refunds on orders older than 180 days require a written exception.

## Audit
Every refund records the approving manager (pseudonymized), the acting agent, the ticket id, and the reason.
