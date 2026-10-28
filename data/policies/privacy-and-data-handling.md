# Privacy & Data Handling (internal)

## Principle
Support agents act on behalf of a user and are limited to that user's data. A customer token can only read the customer's own orders. Staff tokens can read any order within their store (tenant). No token can read across stores.

## PII
Names, emails, phone numbers, and payment details are never written to logs, traces, or tickets. The redaction layer replaces them with typed placeholders such as `<EMAIL_ADDRESS>`. Order numbers and SKUs are not PII and may be logged.

## Identity logging
Every tool call records two identities separately: the software agent that made the call and the user it acted for. The user identity is stored as a pseudonymous hash.

## Retention
Audit logs are kept for 13 months. Chat transcripts are kept for 90 days.
