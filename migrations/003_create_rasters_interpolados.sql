CREATE TABLE IF NOT EXISTS rasters_interpolados (
    id BIGSERIAL PRIMARY KEY,
    tipo TEXT NOT NULL CHECK (tipo IN ('talhao', 'gleba')),
    identificador INTEGER NOT NULL,
    processo TEXT NOT NULL CHECK (processo IN ('solo', 'foliar', 'compac', 'nemat', 'prod')),
    atributo TEXT NOT NULL,
    campanha TEXT,
    url TEXT NOT NULL,
    data DATE NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    cliente INTEGER NOT NULL,
    fazenda INTEGER NOT NULL,
    talhao INTEGER NOT NULL,
    gleba INTEGER,
    profundidade INTEGER NOT NULL
);
