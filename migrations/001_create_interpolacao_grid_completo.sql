CREATE TABLE IF NOT EXISTS interpolacao_grid_completo (
    id BIGSERIAL PRIMARY KEY,
    tipo TEXT NOT NULL CHECK (tipo IN ('talhao', 'gleba')),
    identificador INTEGER NOT NULL,
    id_ponto INTEGER NOT NULL,
    processo TEXT NOT NULL CHECK (processo IN ('solo', 'foliar', 'compac', 'nemat', 'prod')),
    cliente INTEGER NOT NULL,
    fazenda INTEGER NOT NULL,
    talhao INTEGER NOT NULL,
    gleba INTEGER,
    profundidade INTEGER NOT NULL,
    data DATE NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    long DOUBLE PRECISION NOT NULL,
    geometry GEOGRAPHY(POINT, 4326)
);
