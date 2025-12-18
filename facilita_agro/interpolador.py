"""
interpolador.py - Interpolação espacial para atributos de amostragem
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import re
import unicodedata

import numpy as np
import geopandas as gpd
from shapely.geometry import mapping

from .logger import LoggerAgricola, NivelLog

# ----------------------------------------------------------------------
# Dependências opcionais
# ----------------------------------------------------------------------

try:
    import rasterio
    from rasterio.transform import from_origin, Affine
    from rasterio.features import geometry_mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from pykrige.ok import OrdinaryKriging
    HAS_PYKRIGE = True
except ImportError:
    HAS_PYKRIGE = False


class InterpoladorAtributo:
    """
    Responsável por:
      - criar grade raster 10 x 10 m (ou outro pixel_size)
      - interpolar atributos por IDW ou Kriging
      - mascarar e recortar pelo contorno da lavoura
      - salvar GeoTIFF
    """

    def __init__(
        self,
        contorno_utm: gpd.GeoDataFrame,
        crs_utm: str,
        logger: Optional[LoggerAgricola] = None,
        pixel_size: float = 10.0,
        nodata: float = -9999.0,
    ) -> None:
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio não está disponível. Instale para usar o InterpoladorAtributo.")

        self.contorno_utm = contorno_utm
        self.crs_utm = crs_utm
        self.logger = logger
        self.pixel_size = float(pixel_size)
        self.nodata = float(nodata)

        self.geom_union = self.contorno_utm.geometry.unary_union

        if not HAS_SCIPY:
            self._log(
                NivelLog.WARNING,
                "init_interpolador",
                "scipy não encontrado. IDW ficará indisponível.",
            )
        if not HAS_PYKRIGE:
            self._log(
                NivelLog.WARNING,
                "init_interpolador",
                "pykrige não encontrado. Kriging ficará indisponível.",
            )

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log(
        self,
        nivel: NivelLog,
        etapa: str,
        mensagem: str,
        dados: Optional[Dict[str, Any]] = None,
        mostrar_usuario: bool = False,
    ) -> None:
        if self.logger is not None:
            self.logger.log(nivel, etapa, mensagem, dados, mostrar_usuario)
        else:
            print(f"[{nivel.value}] {etapa}: {mensagem}")


    # ------------------------------------------------------------------
    # value_cols helper
    # ------------------------------------------------------------------
    def _sanitizar_nome(self, texto: str) -> str:
        """
        Gera um nome 'seguro' para arquivo a partir de um rótulo de coluna.

        - remove acentos
        - converte não alfanumérico em '_'
        - comprime múltiplos '_' em um só
        - remove '_' do começo/fim
        """
        s = unicodedata.normalize("NFKD", texto)
        s = s.encode("ascii", "ignore").decode("ascii")

        s = re.sub(r"[^A-Za-z0-9]+", "_", s)

        s = re.sub(r"_+", "_", s).strip("_")

        return s


    # ------------------------------------------------------------------
    # Grade raster
    # ------------------------------------------------------------------

    def _criar_grade(self) -> Tuple[np.ndarray, np.ndarray, rasterio.Affine]:
        """
        Cria grade de centros de pixels (xs, ys) e o transform affine
        com base no bounding box do contorno.
        """
        minx, miny, maxx, maxy = self.contorno_utm.total_bounds
        px = self.pixel_size

        xs = np.arange(minx + px / 2, maxx, px)
        ys = np.arange(maxy - px / 2, miny, -px)

        transform = from_origin(minx, maxy, px, px)

        self._log(
            NivelLog.DEBUG,
            "criar_grade",
            f"Grade criada: {len(xs)} colunas x {len(ys)} linhas (pixel={px}m).",
        )

        return xs, ys, transform

    # ------------------------------------------------------------------
    # IDW
    # ------------------------------------------------------------------

    def _interpolar_idw(
        self,
        gdf_utm: gpd.GeoDataFrame,
        value_col: str,
        power: float = 2.0,
        k: int = 12,
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Interpolação IDW (Inverse Distance Weighting) usando k vizinhos mais próximos.
        Retorna (grid, transform).
        """
        if not HAS_SCIPY:
            raise RuntimeError("scipy não disponível para IDW.")

        xs, ys, transform = self._criar_grade()
        px = self.pixel_size

        gdf_valid = gdf_utm.dropna(subset=[value_col]).copy()
        x = gdf_valid.geometry.x.values
        y = gdf_valid.geometry.y.values
        z = gdf_valid[value_col].values

        n = len(z)
        if n == 0:
            raise RuntimeError(f"Nenhum valor válido em '{value_col}' para IDW.")

        # Caso especial: apenas 1 ponto → raster constante
        if n == 1:
            grid_vals = np.full((len(ys), len(xs)), float(z[0]), dtype=float)
            self._log(
                NivelLog.INFO,
                "idw",
                f"Apenas 1 ponto válido em '{value_col}'. "
                f"Raster IDW degenerado para valor constante ({z[0]}).",
            )
            return grid_vals, transform

        # Para n >= 2, segue o IDW normal
        if n < k:
            k = n

        tree = cKDTree(np.column_stack([x, y]))

        grid_x, grid_y = np.meshgrid(xs, ys)  
        pontos_grade = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        dist, idx = tree.query(pontos_grade, k=k)

        dist = np.asarray(dist)
        idx = np.asarray(idx)
        if dist.ndim == 1:
            dist = dist[:, np.newaxis]
            idx = idx[:, np.newaxis]

        with np.errstate(divide="ignore"):
            weights = 1.0 / (dist ** power)

        weights[~np.isfinite(weights)] = 0.0

        z_neighbors = z[idx]
        num = np.sum(weights * z_neighbors, axis=1)
        den = np.sum(weights, axis=1)

        grid_vals = np.where(den > 0, num / den, np.nan)
        grid_vals = grid_vals.reshape(grid_x.shape)

        self._log(
            NivelLog.INFO,
            "idw",
            f"IDW concluído para '{value_col}' com {n} pontos válidos.",
        )

        return grid_vals, transform


    # ------------------------------------------------------------------
    # Kriging
    # ------------------------------------------------------------------

    def _interpolar_kriging(
        self,
        gdf_utm: gpd.GeoDataFrame,
        value_col: str,
        variogram_model: str = "spherical",
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Interpolação por Ordinary Kriging usando pykrige.
        Retorna (grid, transform).
        """
        if not HAS_PYKRIGE:
            raise RuntimeError("pykrige não disponível para Kriging.")

        xs, ys_desc, transform = self._criar_grade()

        gdf_valid = gdf_utm.dropna(subset=[value_col]).copy()
        x = gdf_valid.geometry.x.values
        y = gdf_valid.geometry.y.values
        z = gdf_valid[value_col].values

        n = len(z)
        if n < 10:
            raise RuntimeError(f"Amostra muito pequena para Kriging (n={n}).")

        ys_asc = np.sort(ys_desc) 

        OK = OrdinaryKriging(
            x,
            y,
            z,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
        )

        z_krig, ss = OK.execute("grid", xs, ys_asc)

        z_krig = np.array(z_krig)
        grid_vals = z_krig[::-1, :]  

        self._log(
            NivelLog.INFO,
            "kriging",
            f"Kriging ({variogram_model}) concluído para '{value_col}' com {n} pontos válidos.",
        )

        return grid_vals, transform

    # ------------------------------------------------------------------
    # Máscara pelo contorno e recorte
    # ------------------------------------------------------------------

    def _mascarar_e_recortar(
        self,
        grid: np.ndarray,
        transform: rasterio.Affine,
    ) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Aplica máscara do contorno da lavoura e recorta o raster ao menor
        bounding box que contém a área da lavoura.
        """
        mask_inside = ~geometry_mask(
            [mapping(self.geom_union)],
            out_shape=grid.shape,
            transform=transform,
            invert=True,  
        )

        grid_masked = np.where(mask_inside, np.nan, grid)

        valid = np.isfinite(grid_masked)
        if not valid.any():
            self._log(
                NivelLog.WARNING,
                "mascarar_recortar",
                "Nenhum valor válido após máscara pelo contorno.",
            )
            return grid_masked, transform

        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)

        row_idx = np.where(rows)[0]
        col_idx = np.where(cols)[0]

        r_min, r_max = row_idx[0], row_idx[-1]
        c_min, c_max = col_idx[0], col_idx[-1]

        grid_cropped = grid_masked[r_min : r_max + 1, c_min : c_max + 1]

        offset_transform = Affine.translation(c_min, r_min)
        new_transform = transform * offset_transform

        self._log(
            NivelLog.DEBUG,
            "mascarar_recortar",
            f"Raster recortado: linhas {r_min}-{r_max}, colunas {c_min}-{c_max}.",
        )

        return grid_cropped, new_transform

    # ------------------------------------------------------------------
    # Salvar GeoTIFF
    # ------------------------------------------------------------------

    def _salvar_geotiff(
        self,
        caminho_saida: Path,
        grid: np.ndarray,
        transform: rasterio.Affine,
    ) -> None:
        """
        Salva o raster interpolado em GeoTIFF (1 banda).
        """
        caminho_saida.parent.mkdir(parents=True, exist_ok=True)

        height, width = grid.shape

        data = np.where(np.isfinite(grid), grid, self.nodata).astype("float32")

        with rasterio.open(
            caminho_saida,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=self.crs_utm,
            transform=transform,
            nodata=self.nodata,
        ) as dst:
            dst.write(data, 1)

        self._log(
            NivelLog.INFO,
            "salvar_geotiff",
            f"Raster salvo em: {caminho_saida}",
        )

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def interpolar_atributo(
        self,
        gdf_utm: gpd.GeoDataFrame,
        value_col: str,
        metodo: str,
        nome_lavoura: str,
        ano_amostragem: str,
        diretorio_saida: str | Path,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Executa a interpolação para um atributo específico.

        Args:
            gdf_utm: GeoDataFrame de pontos em UTM.
            value_col: nome da coluna a interpolar.
            metodo: 'idw' ou 'kriging' (se kriging indisponível, cai para IDW).
            nome_lavoura: usado no nome do arquivo.
            ano_amostragem: string de 4 dígitos.
            diretorio_saida: pasta onde salvar o GeoTIFF.

        Retorna:
            (sucesso, mensagem, caminho_arquivo_ou_None)
        """
        diretorio_saida = Path(diretorio_saida)

        nome_col_sanit = self._sanitizar_nome(value_col)
        lavoura_sanit = self._sanitizar_nome(str(nome_lavoura))
        campanha_sanit = self._sanitizar_nome(str(ano_amostragem))

        nome_base = f"{lavoura_sanit}_{nome_col_sanit}_{campanha_sanit}"
        caminho_tif = diretorio_saida / f"{nome_base}.tif"

        try:
            if metodo.lower() == "kriging":
                if not HAS_PYKRIGE:
                    self._log(
                        NivelLog.WARNING,
                        "interpolar_atributo",
                        f"pykrige indisponível. Atributo '{value_col}' será interpolado por IDW.",
                        mostrar_usuario=True,
                    )
                    grid, transform = self._interpolar_idw(gdf_utm, value_col)
                else:
                    grid, transform = self._interpolar_kriging(gdf_utm, value_col)
            else:
                if not HAS_SCIPY:
                    raise RuntimeError("scipy indisponível para IDW.")
                grid, transform = self._interpolar_idw(gdf_utm, value_col)

            grid_masked, new_transform = self._mascarar_e_recortar(grid, transform)

            self._salvar_geotiff(caminho_tif, grid_masked, new_transform)

            msg = f"Atributo '{value_col}' interpolado por {metodo.upper()} e salvo em {caminho_tif.name}"
            self._log(NivelLog.INFO, "interpolar_atributo", msg, mostrar_usuario=True)
            return True, msg, caminho_tif

        except Exception as e:
            msg = f"Erro ao interpolar atributo '{value_col}' ({metodo}): {e}"
            self._log(NivelLog.ERROR, "interpolar_atributo", msg, mostrar_usuario=True)
            return False, msg, None
