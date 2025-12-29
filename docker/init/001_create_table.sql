DROP TABLE IF EXISTS embeddings;

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    img_name TEXT NOT NULL,
    face BYTEA NOT NULL,
    face_shape INT[] NOT NULL,
    model_name TEXT NOT NULL,
    detector_backend TEXT NOT NULL,
    aligned BOOLEAN DEFAULT true,
    l2_normalized BOOLEAN  DEFAULT false,
    embedding FLOAT8[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),

    face_hash TEXT NOT NULL,
    embedding_hash TEXT NOT NULL,

    UNIQUE (face_hash, embedding_hash)
);

CREATE TABLE IF NOT EXISTS embeddings_index (
    id SERIAL PRIMARY KEY,
    model_name TEXT,
    detector_backend TEXT,
    align BOOL,
    l2_normalized BOOL,
    index_data BYTEA,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE (model_name, detector_backend, align, l2_normalized)
);