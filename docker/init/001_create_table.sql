DROP TABLE IF EXISTS embeddings;

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    img_name TEXT NOT NULL,
    face JSONB NOT NULL,
    model_name TEXT NOT NULL,
    detector_backend TEXT NOT NULL,
    aligned BOOLEAN DEFAULT true,
    l2_normalized BOOLEAN  DEFAULT false,
    embedding FLOAT8[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),

    face_hash BYTEA NOT NULL,
    embedding_hash BYTEA NOT NULL,

    UNIQUE (face_hash, embedding_hash)
);