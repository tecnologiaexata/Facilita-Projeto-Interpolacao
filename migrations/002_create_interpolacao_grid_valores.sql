CREATE TABLE IF NOT EXISTS interpolacao_grid_valores (
    id BIGSERIAL PRIMARY KEY,
    grid_completo_id BIGINT NOT NULL REFERENCES interpolacao_grid_completo(id) ON DELETE CASCADE,
    processo TEXT NOT NULL CHECK (processo IN ('solo', 'foliar', 'compac', 'nemat', 'prod')),
    coluna TEXT NOT NULL,
    valor NUMERIC
);
