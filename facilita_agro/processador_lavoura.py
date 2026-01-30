"""
processador_lavoura.py - Gestão de contorno e grade amostral da lavoura
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import geopandas as gpd
import pandas as pd
import numpy as np
import requests
from shapely.geometry import Point
from shapely.ops import nearest_points

from .logger import LoggerAgricola, NivelLog

warnings.filterwarnings("ignore")


class ProcessadorLavoura:
    """
    Responsável por:
      - carregar e validar o contorno da lavoura (KML → UTM)
      - criar ou gerenciar a grade amostral de 1 ha
      - converter grades antigas de polígonos para pontos
      - aplicar filtro de distância mínima à borda
    """

    def __init__(
        self,
        nome_lavoura: str,
        diretorio_base: str | Path,
        logger: Optional[LoggerAgricola] = None,
    ) -> None:
        self.nome_lavoura = nome_lavoura
        self.diretorio_base = Path(diretorio_base)
        self.logger = logger

        self.contorno: Optional[gpd.GeoDataFrame] = None
        self.grade: Optional[gpd.GeoDataFrame] = None
        self.crs_utm: Optional[str] = None
        self.tipo: Optional[str] = None
        self.identificador: Optional[int] = None
        self.url_kml: Optional[str] = None
        self.url_grade: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _log(
        self,
        nivel: NivelLog,
        etapa: str,
        mensagem: str,
        dados: Optional[Dict[str, Any]] = None,
        mostrar_usuario: bool = False,
    ) -> None:
        """Wrapper para logging opcional."""
        if self.logger is not None:
            self.logger.log(nivel, etapa, mensagem, dados, mostrar_usuario)
        else:
            print(f"[{nivel.value}] {etapa}: {mensagem}")

    def _baixar_arquivo(self, url: str, sufixo_padrao: str) -> Path:
        """Baixa um arquivo remoto e retorna o caminho temporário."""
        resposta = requests.get(url, timeout=60)
        resposta.raise_for_status()

        sufixo = Path(url).suffix or sufixo_padrao
        with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as arquivo_temp:
            arquivo_temp.write(resposta.content)
            return Path(arquivo_temp.name)

    def _consultar_kml_grade(self, tipo: str, identificador: int) -> Dict[str, Any]:
        """Consulta o endpoint do front para obter URLs do KML e grade."""
        base_url = os.getenv("FACILITAGRO_FRONTEND_BASE_URL", "https://facilitagro.com.br")
        endpoint = f"{base_url.rstrip('/')}/api/v2/consultarKmlGrade"
        resposta = requests.post(
            endpoint,
            json={"tipo": tipo, "id": identificador},
            timeout=30,
        )
        resposta.raise_for_status()
        dados = resposta.json()

        if "url_kml" not in dados:
            raise ValueError("Resposta de consultarKmlGrade sem url_kml.")

        return dados

    def _atualizar_kml_grade(self, tipo: str, identificador: int, url_grade: str) -> None:
        """Envia ao front a URL pública da grade para correlação."""
        base_url = os.getenv("FACILITAGRO_FRONTEND_BASE_URL", "https://facilitagro.com.br")
        endpoint = f"{base_url.rstrip('/')}/api/v2/atualizarKmlGrade"
        resposta = requests.post(
            endpoint,
            json={"tipo": tipo, "id": identificador, "url_grade": url_grade},
            timeout=30,
        )
        resposta.raise_for_status()

    def _upload_grade_blob(self, caminho_grade: Path, nome_arquivo: str) -> str:
        """Envia a grade para o blob storage da Vercel e retorna a URL pública."""
        token = os.getenv("NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN")
        if not token:
            raise ValueError("Variável NEXT_PUBLIC_BLOB_READ_WRITE_TOKEN não configurada.")

        endpoint = f"https://blob.vercel-storage.com/{nome_arquivo}"
        with open(caminho_grade, "rb") as arquivo:
            resposta = requests.put(
                endpoint,
                data=arquivo,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/octet-stream",
                },
                timeout=60,
            )
        resposta.raise_for_status()

        url_blob = resposta.headers.get("x-vercel-blob-url")
        if url_blob:
            return url_blob

        try:
            dados = resposta.json()
        except ValueError:
            dados = {}

        return dados.get("url") or dados.get("downloadUrl") or endpoint

    # ------------------------------------------------------------------
    # 1. Contorno
    # ------------------------------------------------------------------

    def checar_perimetro(self) -> Tuple[bool, str]:
        """
        Verifica e importa o perímetro da lavoura (KML em 4326 → UTM).
        """
        arquivo_contorno = self.diretorio_base / f"{self.nome_lavoura}_contorno.kml"

        if not arquivo_contorno.exists():
            msg = f"Arquivo de contorno não encontrado: {arquivo_contorno}"
            self._log(NivelLog.WARNING, "checar_perimetro", msg, mostrar_usuario=True)
            return False, "Arquivo de contorno não encontrado"

        try:
            gdf = gpd.read_file(arquivo_contorno)

            if gdf.crs is None:
                gdf = gdf.set_crs(4326)

            centroid = gdf.to_crs(4326).geometry.centroid.iloc[0]
            zona_utm = self._determinar_zona_utm(centroid.y, centroid.x)

            gdf_utm = gdf.to_crs(zona_utm)

            self.contorno = gdf_utm
            self.crs_utm = zona_utm

            msg = f"Perímetro carregado com sucesso. CRS: {zona_utm}"
            self._log(NivelLog.INFO, "checar_perimetro", msg, mostrar_usuario=True)
            return True, msg

        except Exception as e:
            msg = f"Erro ao carregar perímetro: {str(e)}"
            self._log(NivelLog.ERROR, "checar_perimetro", msg, mostrar_usuario=True)
            return False, msg

    def checar_perimetroV2(self, tipo: str, identificador: int) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Verifica e importa o perímetro da lavoura via URL de KML (KML em 4326 → UTM).
        Consulta o endpoint do front com tipo/id para obter url_kml/url_grade.
        """
        try:
            dados = self._consultar_kml_grade(tipo, identificador)
            url_kml = dados.get("url_kml")
            url_grade = dados.get("url_grade")

            if not url_kml:
                msg = "URL do KML não encontrada."
                self._log(NivelLog.WARNING, "checar_perimetroV2", msg, mostrar_usuario=True)
                return False, msg, dados

            caminho_kml = self._baixar_arquivo(url_kml, ".kml")
            gdf = gpd.read_file(caminho_kml)
            caminho_kml.unlink(missing_ok=True)

            if gdf.crs is None:
                gdf = gdf.set_crs(4326)

            centroid = gdf.to_crs(4326).geometry.centroid.iloc[0]
            zona_utm = self._determinar_zona_utm(centroid.y, centroid.x)

            gdf_utm = gdf.to_crs(zona_utm)

            self.contorno = gdf_utm
            self.crs_utm = zona_utm
            self.tipo = tipo
            self.identificador = identificador
            self.url_kml = url_kml
            self.url_grade = url_grade

            msg = f"Perímetro carregado com sucesso. CRS: {zona_utm}"
            self._log(NivelLog.INFO, "checar_perimetroV2", msg, mostrar_usuario=True)

            return True, msg, {
                "tipo": tipo,
                "id": identificador,
                "url_kml": url_kml,
                "url_grade": url_grade,
            }

        except Exception as e:
            msg = f"Erro ao carregar perímetro: {str(e)}"
            self._log(NivelLog.ERROR, "checar_perimetroV2", msg, mostrar_usuario=True)
            return False, msg, {"tipo": tipo, "id": identificador}

    def _determinar_zona_utm(self, lat: float, lon: float) -> str:
        """
        Determina a zona UTM automaticamente baseada nas coordenadas (lat, lon).
        Retorna string do tipo 'EPSG:32722'.
        """
        zona = int((lon + 180) // 6) + 1
        hemisferio = 32600 if lat >= 0 else 32700
        return f"EPSG:{hemisferio + zona}"

    # ------------------------------------------------------------------
    # 2. Grade (gerenciar / criar)
    # ------------------------------------------------------------------

    def gerenciar_grade(self) -> Tuple[bool, str]:
        """
        Verifica se grade existe; se não, cria uma nova de ~1 ha.
        Sempre produz grade PONTUAL, com colunas:
          - id_ponto
          - geometry
          - area_representada_ha
        """
        arquivo_grade = self.diretorio_base / f"{self.nome_lavoura}_grade.gpkg"

        if arquivo_grade.exists():
            try:
                gdf_grade = gpd.read_file(arquivo_grade)
                gdf_grade = self._validar_e_corrigir_grade(gdf_grade)
                self.grade = gdf_grade

                msg = f"Grade existente carregada: {len(self.grade)} pontos"
                self._log(NivelLog.INFO, "gerenciar_grade", msg, mostrar_usuario=True)
                return True, msg
            except Exception as e:
                msg = f"Erro ao carregar grade: {str(e)}"
                self._log(NivelLog.ERROR, "gerenciar_grade", msg, mostrar_usuario=True)
                return False, msg

        if self.contorno is None:
            msg = "Perímetro não carregado. Execute checar_perimetro() primeiro."
            self._log(NivelLog.ERROR, "gerenciar_grade", msg, mostrar_usuario=True)
            return False, msg

        return self._criar_grade_1ha(arquivo_grade)

    def gerenciar_gradeV2(
        self,
        tipo: str,
        identificador: int,
        url_kml: str,
        url_grade: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Gerencia grade via URL. Se url_grade existir, valida; senão cria, envia ao blob
        storage e atualiza o endpoint do front.
        """
        try:
            if self.contorno is None:
                msg = "Perímetro não carregado. Execute checar_perimetroV2() primeiro."
                self._log(NivelLog.ERROR, "gerenciar_gradeV2", msg, mostrar_usuario=True)
                return False, msg, {
                    "tipo": tipo,
                    "id": identificador,
                    "url_kml": url_kml,
                    "url_grade": url_grade,
                }

            if url_grade:
                caminho_grade = self._baixar_arquivo(url_grade, ".gpkg")
                gdf_grade = gpd.read_file(caminho_grade)
                caminho_grade.unlink(missing_ok=True)
                gdf_grade = self._validar_e_corrigir_grade(gdf_grade)
                self.grade = gdf_grade

                msg = f"Grade existente carregada: {len(self.grade)} pontos"
                self._log(NivelLog.INFO, "gerenciar_gradeV2", msg, mostrar_usuario=True)
                return True, msg, {
                    "tipo": tipo,
                    "id": identificador,
                    "url_kml": url_kml,
                    "url_grade": url_grade,
                }

            with tempfile.TemporaryDirectory() as temp_dir:
                caminho_grade = Path(temp_dir) / f"{tipo}_{identificador}_grade.gpkg"
                ok, msg = self._criar_grade_1ha(caminho_grade)
                if not ok:
                    return False, msg, {
                        "tipo": tipo,
                        "id": identificador,
                        "url_kml": url_kml,
                        "url_grade": url_grade,
                    }

                nome_arquivo = f"{tipo}_{identificador}_grade.gpkg"
                url_grade_publica = self._upload_grade_blob(caminho_grade, nome_arquivo)
                self._atualizar_kml_grade(tipo, identificador, url_grade_publica)

            self.url_grade = url_grade_publica

            msg = "Grade criada, enviada ao blob storage e vinculada com sucesso."
            self._log(NivelLog.INFO, "gerenciar_gradeV2", msg, mostrar_usuario=True)
            return True, msg, {
                "tipo": tipo,
                "id": identificador,
                "url_kml": url_kml,
                "url_grade": url_grade_publica,
            }

        except Exception as e:
            msg = f"Erro ao gerenciar grade: {str(e)}"
            self._log(NivelLog.ERROR, "gerenciar_gradeV2", msg, mostrar_usuario=True)
            return False, msg, {
                "tipo": tipo,
                "id": identificador,
                "url_kml": url_kml,
                "url_grade": url_grade,
            }

    def _validar_e_corrigir_grade(self, gdf_grade: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Garante que a grade PONTUAL tem todas as colunas necessárias.
        Se for uma grade antiga de polígonos, converte para pontos.
        """
        if len(gdf_grade) == 0:
            return gdf_grade

        if gdf_grade.geometry.iloc[0].geom_type == "Polygon":
            self._log(
                NivelLog.INFO,
                "validar_grade",
                "Convertendo grade de polígonos para grade pontual...",
            )
            return self._converter_poligonos_para_pontos(gdf_grade)

        if "id_ponto" not in gdf_grade.columns:
            gdf_grade["id_ponto"] = range(len(gdf_grade))

        if "area_representada_ha" not in gdf_grade.columns and self.contorno is not None:
            area_total = self.contorno.geometry.area.sum() / 10_000
            n_pontos = len(gdf_grade)
            gdf_grade["area_representada_ha"] = area_total / n_pontos if n_pontos > 0 else 0

        return gdf_grade

    def _converter_poligonos_para_pontos(
        self,
        gdf_poligonos: gpd.GeoDataFrame,
        distancia_borda: float = 15.0,
    ) -> gpd.GeoDataFrame:
        """
        Converte grade de polígonos antiga para grade pontual com filtro de borda.
        Usa centróides dos polígonos + filtro de distância mínima ao contorno.
        """
        if self.crs_utm is None:
            raise ValueError("CRS UTM não definido. Carregue o perímetro primeiro.")

        pontos = gdf_poligonos.geometry.centroid
        gdf_pontos = gpd.GeoDataFrame(
            {"id_ponto": range(len(pontos)), "geometry": pontos},
            crs=self.crs_utm,
        )

        geom_union = self.contorno.geometry.unary_union
        gdf_dentro = gdf_pontos[gdf_pontos.within(geom_union)].copy()
        gdf_dentro = gdf_dentro.reset_index(drop=True)
        gdf_dentro["id_ponto"] = range(len(gdf_dentro))

        gdf_final = self._filtrar_pontos_afastados_bordas(gdf_dentro, distancia_borda)
        gdf_final = gdf_final.reset_index(drop=True)
        gdf_final["id_ponto"] = range(len(gdf_final))

        area_total = self.contorno.geometry.area.sum() / 10_000
        n_pontos_final = len(gdf_final)
        gdf_final["area_representada_ha"] = (
            area_total / n_pontos_final if n_pontos_final > 0 else 0
        )

        return gdf_final

    def _filtrar_pontos_afastados_bordas(
        self,
        gdf_points: gpd.GeoDataFrame,
        distancia_minima: float = 15.0,
    ) -> gpd.GeoDataFrame:
        """
        Move pontos para o interior de um buffer interno se estiverem muito próximos da borda.
        """
        try:
            contorno_buffer = self.contorno.geometry.buffer(-distancia_minima)

            buffer_union = contorno_buffer.unary_union

            mascara_fora_buffer = ~gdf_points.within(buffer_union)
            pontos_fora_buffer = gdf_points[mascara_fora_buffer].copy()
            pontos_dentro_buffer = gdf_points[~mascara_fora_buffer].copy()

            self._log(
                NivelLog.INFO,
                "filtro_borda",
                f"Pontos fora do buffer interno: {len(pontos_fora_buffer)}",
            )

            if len(pontos_fora_buffer) == 0:
                return gdf_points

            pontos_corrigidos: List[Point] = []
            for _, ponto_row in pontos_fora_buffer.iterrows():
                ponto_original = ponto_row.geometry
                ponto_corrigido = self._encontrar_ponto_mais_proximo_no_buffer(
                    ponto_original, contorno_buffer
                )
                pontos_corrigidos.append(ponto_corrigido)

            gdf_corrigidos = gpd.GeoDataFrame(
                {
                    "id_ponto": pontos_fora_buffer["id_ponto"].values,
                    "geometry": pontos_corrigidos,
                },
                crs=self.crs_utm,
            )

            gdf_final = pd.concat(
                [pontos_dentro_buffer, gdf_corrigidos], ignore_index=True
            )
            gdf_final = gdf_final.sort_values("id_ponto").reset_index(drop=True)

            pontos_ainda_fora = gdf_final[~gdf_final.within(buffer_union)]
            success_rate = (
                (len(pontos_fora_buffer) - len(pontos_ainda_fora))
                / len(pontos_fora_buffer)
                * 100
            )

            self._log(
                NivelLog.INFO,
                "filtro_borda",
                f"Taxa de sucesso do reposicionamento: {success_rate:.1f}%",
                dados={
                    "reposicionados": len(pontos_fora_buffer),
                    "ainda_fora": len(pontos_ainda_fora),
                },
            )

            return gdf_final

        except Exception as e:
            self._log(
                NivelLog.ERROR,
                "filtro_borda",
                f"Erro no reposicionamento: {e}",
            )
            return gdf_points

    def _encontrar_ponto_mais_proximo_no_buffer(
        self,
        ponto: Point,
        contorno_buffer,
    ) -> Point:
        """
        Encontra o ponto mais próximo dentro do buffer para um ponto dado.
        Aceita contorno_buffer como GeoSeries ou GeoDataFrame.
        """
        try:
            if hasattr(contorno_buffer, "geometry"):
                geom_buffer = contorno_buffer.geometry.unary_union
            else:
                geom_buffer = contorno_buffer.unary_union

            ponto_buffer, _ = nearest_points(geom_buffer, ponto)
            return ponto_buffer

        except Exception:
            try:
                if hasattr(contorno_buffer, "geometry"):
                    geom_buffer = contorno_buffer.geometry.unary_union
                else:
                    geom_buffer = contorno_buffer.unary_union

                borda = geom_buffer.boundary
                ponto_proj = borda.interpolate(borda.project(ponto))
                return ponto_proj
            except Exception:
                centroide = self.contorno.geometry.unary_union.centroid
                return centroide

    def _criar_grade_1ha(
        self,
        caminho_saida: Path,
        distancia_borda: float = 15.0,
    ) -> Tuple[bool, str]:
        """
        Cria grade PONTUAL regular com pontos a cada ~1 ha (100m x 100m),
        aplica filtro de borda e salva em GPKG.
        """
        try:
            bounds = self.contorno.total_bounds
            minx, miny, maxx, maxy = bounds

            spacing = 100.0  

            x_coords = np.arange(minx + spacing / 2, maxx, spacing)
            y_coords = np.arange(miny + spacing / 2, maxy, spacing)

            points = [Point(x, y) for x in x_coords for y in y_coords]

            gdf_points = gpd.GeoDataFrame(
                {"id_ponto": range(len(points)), "geometry": points},
                crs=self.crs_utm,
            )

            geom_union = self.contorno.geometry.unary_union
            gdf_dentro = gdf_points[gdf_points.within(geom_union)].copy()
            gdf_dentro = gdf_dentro.reset_index(drop=True)
            gdf_dentro["id_ponto"] = range(len(gdf_dentro))

            self._log(
                NivelLog.INFO,
                "criar_grade",
                f"Pontos dentro do contorno (antes do filtro de borda): {len(gdf_dentro)}",
            )

            gdf_final = self._filtrar_pontos_afastados_bordas(
                gdf_dentro, distancia_borda
            )

            gdf_final = gdf_final.reset_index(drop=True)
            gdf_final["id_ponto"] = range(len(gdf_final))

            area_total = self.contorno.geometry.area.sum() / 10_000
            n_pontos_final = len(gdf_final)
            gdf_final["area_representada_ha"] = (
                area_total / n_pontos_final if n_pontos_final > 0 else 0
            )

            self._log(
                NivelLog.INFO,
                "criar_grade",
                (
                    f"Área total: {area_total:.2f} ha | "
                    f"Pontos finais: {n_pontos_final} | "
                    f"Densidade: {n_pontos_final/area_total:.2f} pontos/ha "
                    f"| Distância mínima da borda: {distancia_borda} m"
                ),
                mostrar_usuario=True,
            )

            gdf_final.to_file(caminho_saida, driver="GPKG", layer="grade_pontual")

            self.grade = gdf_final

            msg = (
                f"Grade pontual criada com {len(gdf_final)} pontos (~{area_total:.1f} ha)"
            )
            return True, msg

        except Exception as e:
            msg = f"Erro ao criar grade pontual: {str(e)}"
            self._log(NivelLog.ERROR, "criar_grade", msg, mostrar_usuario=True)
            return False, msg

    # ------------------------------------------------------------------
    # 3. Fluxo principal 
    # ------------------------------------------------------------------

    def preparar_base_amostragem(self) -> Tuple[bool, str]:
        """
        Fluxo principal da etapa de suporte:
          1. Checar perímetro
          2. Gerenciar/ criar grade
        """
        ok, msg = self.checar_perimetro()
        if not ok:
            return False, msg

        ok, msg_grade = self.gerenciar_grade()
        if not ok:
            return False, msg_grade

        return True, f"Base preparada com {len(self.grade)} pontos"

    def visualizar_reposicionamento(self, distancia_borda: float = 15.0) -> None:
        """
        Visualiza o efeito do reposicionamento dos pontos (antes/depois).
        Apenas para uso exploratório em notebook.
        """
        try:
            import matplotlib.pyplot as plt

            if self.contorno is None:
                raise ValueError("Perímetro não carregado.")
            if self.grade is None:
                raise ValueError("Grade não carregada/criada.")

            bounds = self.contorno.total_bounds
            spacing = 100.0
            x_coords = np.arange(bounds[0] + spacing / 2, bounds[2], spacing)
            y_coords = np.arange(bounds[1] + spacing / 2, bounds[3], spacing)

            pontos_originais = [Point(x, y) for x in x_coords for y in y_coords]
            gdf_originais = gpd.GeoDataFrame(
                {"id_ponto": range(len(pontos_originais)), "geometry": pontos_originais},
                crs=self.crs_utm,
            )

            geom_union = self.contorno.geometry.unary_union
            gdf_dentro_original = gdf_originais[
                gdf_originais.within(geom_union)
            ].copy()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            self.contorno.plot(ax=ax1, color="lightgray", edgecolor="black", alpha=0.7)
            gdf_dentro_original.plot(
                ax=ax1, color="red", markersize=40, label="Pontos originais"
            )
            buffer_interno = self.contorno.geometry.buffer(-distancia_borda)
            buffer_interno.plot(
                ax=ax1, color="orange", alpha=0.3, label=f"Zona de exclusão ({distancia_borda}m)"
            )
            ax1.set_title(f"Antes - {len(gdf_dentro_original)} pontos")
            ax1.legend()

            self.contorno.plot(ax=ax2, color="lightgray", edgecolor="black", alpha=0.7)
            self.grade.plot(
                ax=ax2, color="green", markersize=40, label="Pontos reposicionados"
            )
            buffer_interno.plot(
                ax=ax2, color="orange", alpha=0.3, label=f"Zona de exclusão ({distancia_borda}m)"
            )
            ax2.set_title(f"Após - {len(self.grade)} pontos")
            ax2.legend()

            plt.suptitle(
                f"Reposicionamento de Pontos - Distância mínima: {distancia_borda}m",
                fontsize=14,
            )
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self._log(
                NivelLog.ERROR,
                "visualizar_reposicionamento",
                f"Erro na visualização: {e}",
            )
