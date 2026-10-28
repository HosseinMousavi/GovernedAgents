-- Store Support Agent schema. Every business table carries tenant_id; the repository
-- never issues a query without it. pgvector holds the policy/manual chunks.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tenants (
  tenant_id   TEXT PRIMARY KEY,
  name        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
  customer_id TEXT PRIMARY KEY,
  tenant_id   TEXT NOT NULL REFERENCES tenants(tenant_id),
  full_name   TEXT NOT NULL,          -- PII: never selected by tool handlers
  email       TEXT NOT NULL,          -- PII: never selected by tool handlers
  phone       TEXT,                   -- PII
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS customers_tenant_idx ON customers(tenant_id);

CREATE TABLE IF NOT EXISTS products (
  sku             TEXT NOT NULL,
  tenant_id       TEXT NOT NULL REFERENCES tenants(tenant_id),
  name            TEXT NOT NULL,
  category        TEXT NOT NULL,
  price_cents     INTEGER NOT NULL CHECK (price_cents >= 0),
  warranty_months INTEGER NOT NULL DEFAULT 12,
  PRIMARY KEY (tenant_id, sku)
);

CREATE TABLE IF NOT EXISTS inventory (
  tenant_id  TEXT NOT NULL,
  sku        TEXT NOT NULL,
  stock      INTEGER NOT NULL CHECK (stock >= 0),
  PRIMARY KEY (tenant_id, sku),
  FOREIGN KEY (tenant_id, sku) REFERENCES products(tenant_id, sku)
);

CREATE TABLE IF NOT EXISTS orders (
  order_id        INTEGER PRIMARY KEY,
  tenant_id       TEXT NOT NULL REFERENCES tenants(tenant_id),
  customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
  sku             TEXT NOT NULL,
  total_cents     INTEGER NOT NULL CHECK (total_cents >= 0),
  status          TEXT NOT NULL CHECK (status IN ('placed','shipped','delivered','refunded','cancelled')),
  ordered_on      DATE NOT NULL,
  refunded_cents  INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (tenant_id, sku) REFERENCES products(tenant_id, sku)
);
CREATE INDEX IF NOT EXISTS orders_tenant_customer_idx ON orders(tenant_id, customer_id);

CREATE TABLE IF NOT EXISTS tickets (
  ticket_id   TEXT PRIMARY KEY,
  tenant_id   TEXT NOT NULL,
  order_id    INTEGER NOT NULL REFERENCES orders(order_id),
  category    TEXT NOT NULL,
  summary     TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'open',
  created_by  TEXT NOT NULL,          -- agent identity, not user identity
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS appointments (
  appointment_id TEXT PRIMARY KEY,
  tenant_id      TEXT NOT NULL,
  order_id       INTEGER NOT NULL REFERENCES orders(order_id),
  service_date   DATE NOT NULL,
  created_by     TEXT NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS refunds (
  refund_id    TEXT PRIMARY KEY,
  tenant_id    TEXT NOT NULL,
  order_id     INTEGER NOT NULL REFERENCES orders(order_id),
  amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
  reason       TEXT NOT NULL,
  approved_by  TEXT NOT NULL,         -- pseudonymized manager subject
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- RAG store. 1024 dims = BGE-M3 (Ollama). OpenAI text-embedding-3-small is projected to 1024.
CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id   BIGSERIAL PRIMARY KEY,
  tenant_id  TEXT NOT NULL,
  source     TEXT NOT NULL,           -- file name, used for citations
  section    TEXT,
  content    TEXT NOT NULL,
  embedding  vector(1024) NOT NULL
);
CREATE INDEX IF NOT EXISTS doc_chunks_tenant_idx ON doc_chunks(tenant_id);
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx ON doc_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS audit_log (
  id         BIGSERIAL PRIMARY KEY,
  ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
  request_id TEXT NOT NULL,
  agent      TEXT NOT NULL,
  subject    TEXT NOT NULL,           -- pseudonymized
  tenant     TEXT NOT NULL,
  tool       TEXT NOT NULL,
  outcome    TEXT NOT NULL,
  reason     TEXT
);
