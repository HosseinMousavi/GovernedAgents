# Escalation Policy (internal)

## When the knowledge agent escalates to a human
- It cannot find relevant policy text after two retrieval attempts (query rewrite + retry).
- The grader scores the retrieved context below the confidence threshold.
- The customer asks for something the agent is not scoped to do (for example, a customer asking for a refund to be issued directly).
- The customer reports a safety issue (brake failure, battery fire, airbag fault).

## What escalation produces
A ticket with category `other` and a summary that contains no PII, plus a notification to the on-duty staff channel.

## Response targets
Escalations are acknowledged within 1 business hour and resolved within 2 business days.
