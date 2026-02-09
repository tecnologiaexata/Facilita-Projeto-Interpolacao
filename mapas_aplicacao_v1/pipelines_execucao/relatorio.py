"""
relatorio.py

Relatório agronômico (one-page) do mapa de prescrição:
- 1 página por prescrição (PDF paisagem)
- elementos:
  1) mapa colorido + contorno + seta norte + barra de escala
  2) logo empresa (PNG)
  3) caixa de identificação (produtor/fazenda/talhão/safra/produto/áreas/taxas/quantidade total)
  4) legenda (classes discretas pós-arredondamento)

Entrada principal:
- gdf_prescricao (GeoDataFrame) com coluna 'dose' (classes finais) e geometria (polígonos)
- contorno_utm (GeoDataFrame) do talhão (UTM, em metros)

Saída:
- PDF em A4 paisagem
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrow, Patch
from matplotlib.colors import ListedColormap


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class RelatorioConfig:
    # página A4 paisagem em polegadas
    figsize: Tuple[float, float] = (11.69, 8.27)  
    dpi: int = 200

    # layout: fração da largura para a coluna direita
    right_col_frac: float = 0.28

    # texto
    title_text: str = "MAPA"
    title_fontsize: int = 24

    # títulos das caixas (header)
    header_fill_color: str = "#f2f2f2"
    header_fontsize: int = 15

    header_height_fig: float = 0.045  # fração da altura da página A4

    info_title: str = "RESUMO"
    legend_title: str = "LEGENDA"

    info_fontsize: int = 11
    legend_fontsize: int = 11

    # escala
    scalebar_height_frac: float = 0.02
    scalebar_y_offset_frac: float = 0.04
    scalebar_target_frac_width: float = 0.22

    # norte
    north_arrow_size_frac: float = 0.10
    north_arrow_x_frac: float = 0.95
    north_arrow_y_frac: float = 0.08

    # bordas
    border_lw: float = 1.5

    # cores (discretas)
    cmap_name: str = "viridis"

    # unidades
    dose_unit: str = "t/ha"
    dose_decimals: int = 2

    # legenda (layout fixo)
    legend_n_classes: int = 5
    legend_mode: str = "quantiles_area"  # "quantiles_area" | "quantiles" | "equal"
    legend_decimals: Optional[int] = None
    qty_unit: str = "t"

    # fonte global (tenta usar as primeiras disponíveis)
    font_family: str = "sans-serif" 
    font_serif: Tuple[str, ...] = ("Cambria", "Times New Roman", "DejaVu Serif")
    font_sans: Tuple[str, ...] = ("Calibri", "Arial", "DejaVu Sans")

    # logo
    logo_mode: str = "cover"  # "contain" (sem cortar) | "cover" (preenche e pode cortar)


# -------------------------
# Helpers
# -------------------------
def _log(logger, level: str, msg: str):
    if logger is None:
        print(f"[{level}] {msg}")
        return
    fn = getattr(logger, level.lower(), None)
    if callable(fn):
        fn(msg)
    else:
        logger.info(msg)


def _ensure_utm_like(crs) -> None:
    # não trava; só tenta evitar scale bar errada se CRS não estiver em metros
    if crs is None:
        return
    try:
        s = crs.to_string().lower()
        if "epsg:4326" in s:
            pass
    except Exception:
        pass


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not m.any():
        return float("nan")
    return float(np.average(values[m], weights=weights[m]))


def _area_weighted_quantile_bins(values: np.ndarray, areas: np.ndarray, n_classes: int) -> np.ndarray:
    """Quantis ponderados por área. Retorna edges (n_classes+1) de min..max."""
    v = np.asarray(values, dtype=float)
    a = np.asarray(areas, dtype=float)
    m = np.isfinite(v) & np.isfinite(a) & (a > 0)
    v = v[m]
    a = a[m]
    if v.size == 0:
        return np.array([0.0, 1.0])

    order = np.argsort(v)
    v = v[order]
    a = a[order]

    cum = np.cumsum(a)
    total = cum[-1]
    if total <= 0:
        lo = float(np.nanmin(v))
        hi = float(np.nanmax(v))
        return np.linspace(lo, hi, n_classes + 1)

    qs = np.linspace(0, 1, n_classes + 1)
    targets = qs * total
    edges = np.interp(targets, cum, v)
    edges[0] = float(np.nanmin(v))
    edges[-1] = float(np.nanmax(v))

    for i in range(1, len(edges)):
        if edges[i] < edges[i-1]:
            edges[i] = edges[i-1]
    return edges


def _nice_number(x: float) -> float:
    """Arredonda para 1-2-5 * 10^n (padrão para escalas)."""
    if x <= 0:
        return 0.0
    exp = np.floor(np.log10(x))
    f = x / (10 ** exp)
    if f < 1.5:
        nf = 1.0
    elif f < 3.5:
        nf = 2.0
    elif f < 7.5:
        nf = 5.0
    else:
        nf = 10.0
    return float(nf * (10 ** exp))


def _safe_imread(path: str):
    import matplotlib.image as mpimg
    if not path or not os.path.exists(path):
        return None
    try:
        return mpimg.imread(path)
    except Exception:
        return None

def _box_title_fig(fig, ax, title: str, cfg: RelatorioConfig):
    """
    Header com altura fixa na FIGURA (não no eixo),
    garantindo padrão visual em toda a página.
    """
    # bounding box do eixo em coords da figura
    bbox = ax.get_position()

    h = cfg.header_height_fig
    y0 = bbox.y1 - h

    # faixa
    fig.patches.append(
        Rectangle(
            (bbox.x0, y0),
            bbox.width,
            h,
            transform=fig.transFigure,
            facecolor=cfg.header_fill_color,
            edgecolor="black",
            lw=1.0,
            zorder=20,
        )
    )

    # linha inferior do header
    fig.lines.append(
        plt.Line2D(
            [bbox.x0, bbox.x1],
            [y0, y0],
            transform=fig.transFigure,
            lw=1.2,
            color="black",
            zorder=21,
        )
    )

    # título
    fig.text(
        bbox.x0 + 0.01,
        bbox.y1 - 0.01,
        title,
        ha="left",
        va="top",
        fontsize=cfg.header_fontsize,
        fontweight="bold",
        zorder=22,
    )

def _apply_global_font(cfg: RelatorioConfig):
    # fontes melhores em PDF (texto selecionável)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    mpl.rcParams["font.family"] = cfg.font_family
    if cfg.font_family == "serif":
        mpl.rcParams["font.serif"] = list(cfg.font_serif)
    else:
        mpl.rcParams["font.sans-serif"] = list(cfg.font_sans)


def _box_title(ax, title: str, cfg: RelatorioConfig):
    """
    Header preenchido + título no canto superior esquerdo.
    """
    # Faixa superior preenchida
    ax.add_patch(
        Rectangle(
            (0, 0.86), 1, 0.14,
            transform=ax.transAxes,
            facecolor=cfg.header_fill_color,
            edgecolor="black",
            lw=1.0,
            zorder=10,
        )
    )
    # Linha separadora abaixo do header
    ax.add_line(
        plt.Line2D([0, 1], [0.86, 0.86], transform=ax.transAxes, lw=1.2, color="black", zorder=11)
    )

    ax.text(
        0.04, 0.96, title,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=cfg.header_fontsize,
        fontweight="bold",
        zorder=12,
    )


def _draw_logo(ax, logo_path: Optional[str], mode: str = "contain"):
    """
    mode:
      - "contain": encaixa preservando proporção (pode sobrar espaço)
      - "cover": preenche a caixa (pode cortar)
    """
    img = _safe_imread(logo_path) if logo_path else None
    if img is None:
        ax.text(0.5, 0.6, "LOGO EMPRESA\n.PNG", ha="center", va="center", fontsize=16, fontweight="bold")
        return

    h, w = img.shape[0], img.shape[1]
    img_ar = w / h  # largura/altura

    if mode == "contain":
        # Desenha em coords [0..1] preservando proporção
        if img_ar >= 1.0:
            # mais larga: ocupa toda largura
            new_h = 1.0 / img_ar
            y0 = (1.0 - new_h) / 2.0
            extent = (0, 1, y0, y0 + new_h)
        else:
            # mais alta: ocupa toda altura
            new_w = img_ar
            x0 = (1.0 - new_w) / 2.0
            extent = (x0, x0 + new_w, 0, 1)
        ax.imshow(img, extent=extent)
    else:
        # cover: preenche e corta via aspect auto
        ax.imshow(img, extent=(0, 1, 0, 1), aspect="auto")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _draw_north_arrow(ax, cfg: RelatorioConfig):
    x = cfg.north_arrow_x_frac
    y = cfg.north_arrow_y_frac
    size = cfg.north_arrow_size_frac

    arr = FancyArrow(
        x, y, 0, size,
        width=0.03, head_width=0.08, head_length=0.06,
        length_includes_head=True,
        transform=ax.transAxes,
        color="black",
        clip_on=False,
        zorder=20,
    )
    ax.add_patch(arr)
    ax.text(x, y + size + 0.02, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12, fontweight="bold", zorder=21)


def _draw_scalebar(ax, cfg: RelatorioConfig):
    """
    Barra de escala baseada na largura atual do mapa (assumindo CRS em metros).
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width_m = abs(x1 - x0)
    height_m = abs(y1 - y0)
    if width_m <= 0 or height_m <= 0:
        return

    target_len = cfg.scalebar_target_frac_width * width_m
    bar_len = _nice_number(target_len)
    if bar_len <= 0:
        return

    margin_x = 0.03 * width_m
    margin_y = cfg.scalebar_y_offset_frac * height_m

    bar_x1 = max(x0, x1) - margin_x
    bar_x0 = bar_x1 - bar_len
    bar_y0 = min(y0, y1) + margin_y

    bar_h = cfg.scalebar_height_frac * height_m
    if bar_h <= 0:
        bar_h = 0.02 * height_m

    half = bar_len / 2.0

    ax.add_patch(Rectangle((bar_x0, bar_y0), half, bar_h, facecolor="black", edgecolor="black", zorder=20))
    ax.add_patch(Rectangle((bar_x0 + half, bar_y0), half, bar_h, facecolor="white", edgecolor="black", zorder=20))

    ax.text(bar_x0, bar_y0 - 0.6 * bar_h, "0", ha="center", va="top", fontsize=10, zorder=21)
    ax.text(bar_x0 + half, bar_y0 - 0.6 * bar_h, f"{int(round(half))}", ha="center", va="top", fontsize=10, zorder=21)
    ax.text(bar_x1, bar_y0 - 0.6 * bar_h, f"{int(round(bar_len))} m", ha="center", va="top", fontsize=10, zorder=21)


def _discrete_cmap(values_sorted: Sequence[float], cmap_name: str) -> Tuple[ListedColormap, Dict[float, Any]]:
    import matplotlib.cm as cm
    base = cm.get_cmap(cmap_name, len(values_sorted))
    colors = [base(i) for i in range(len(values_sorted))]
    cmap = ListedColormap(colors)
    mapping = {v: colors[i] for i, v in enumerate(values_sorted)}
    return cmap, mapping


def _fmt_ptbr(x: float, nd: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    s = f"{x:,.{nd}f}"
    # 1,234.56 -> 1.234,56
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# -------------------------
# Core
# -------------------------
def gerar_relatorio_prescricao(
    gdf_prescricao: gpd.GeoDataFrame,
    contorno_utm: gpd.GeoDataFrame,
    output_pdf_path: str,
    logo_path: Optional[str],
    info: Dict[str, Any],
    cfg: Optional[RelatorioConfig] = None,
    logger=None,
) -> str:
    """
    Gera 1 PDF (1 página) do mapa.

    info esperado (chaves recomendadas):
      produtor, fazenda, talhao, safra, produto
    """
    cfg = cfg or RelatorioConfig()
    _apply_global_font(cfg)

    if gdf_prescricao is None or len(gdf_prescricao) == 0:
        raise ValueError("gdf_prescricao vazio. Não há prescrição para plotar.")
    if "dose" not in gdf_prescricao.columns:
        raise ValueError("gdf_prescricao precisa ter coluna 'dose' (classes finais).")

    _ensure_utm_like(gdf_prescricao.crs)
    _ensure_utm_like(contorno_utm.crs)


    gdf = gdf_prescricao.copy()
    gdf["dose"] = gdf["dose"].astype(float)

    # áreas (ha)
    if "area_ha" in gdf.columns:
        areas = gdf["area_ha"].astype(float).to_numpy()
    else:
        areas = (gdf.geometry.area / 10000.0).to_numpy()

    doses = gdf["dose"].to_numpy()

    area_total_ha = float(np.nansum(areas))
    dose_min = float(np.nanmin(doses))
    dose_max = float(np.nanmax(doses))
    dose_mean = _weighted_mean(doses, areas)   
    unit = cfg.dose_unit
    if "dose_unit" in gdf.columns:
        try:
            u0 = gdf["dose_unit"].dropna().astype(str).iloc[0]
            if u0:
                unit = u0
        except Exception:
            pass

    decimals = int(getattr(cfg, "dose_decimals", 2))
    if "dose_decimals" in gdf.columns:
        try:
            decimals = int(gdf["dose_decimals"].dropna().iloc[0])
        except Exception:
            pass

    # quantidade total (taxa * área): ex. t, kg, L
    qty_unit = unit.split("/")[0] if "/" in unit else unit
    total_qty = float(np.nansum(doses * areas))

    # classes para mapa/legenda (sempre fixas p/ layout)
    n_cls = int(getattr(cfg, "legend_n_classes", 5))
    mode = str(getattr(cfg, "legend_mode", "quantiles_area")).lower()

    if n_cls < 2:
        n_cls = 2

    if mode in ("quantiles_area", "quantiles-area"):
        edges = _area_weighted_quantile_bins(doses, areas, n_cls)
    elif mode in ("quantiles", "quantil", "quantile"):
        finite = np.isfinite(doses)
        if finite.any():
            edges = np.quantile(doses[finite], np.linspace(0, 1, n_cls + 1))
            edges[0] = float(np.nanmin(doses[finite]))
            edges[-1] = float(np.nanmax(doses[finite]))
        else:
            edges = np.array([0.0, 1.0])
    elif mode in ("equal", "equal_interval", "equal-interval"):
        finite = np.isfinite(doses)
        lo = float(np.nanmin(doses[finite])) if finite.any() else 0.0
        hi = float(np.nanmax(doses[finite])) if finite.any() else 1.0
        edges = np.linspace(lo, hi, n_cls + 1)
    else:
        raise ValueError(f"legend_mode inválido: {mode}")

    internal = edges[1:-1]
    bins = np.digitize(doses, internal, right=True).astype(int)
    bins[~np.isfinite(doses)] = -1

    classes = np.arange(n_cls, dtype=int)
    _, color_map = _discrete_cmap(classes, cfg.cmap_name)

    leg_dec = getattr(cfg, "legend_decimals", None)
    leg_dec = int(leg_dec) if leg_dec is not None else decimals
    labels_5 = [f"{edges[i]:.{leg_dec}f}–{edges[i+1]:.{leg_dec}f} {unit}" for i in range(n_cls)]

    # ---------------- Layout ----------------
    fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)

    # borda geral
    ax_border = fig.add_axes([0, 0, 1, 1])
    ax_border.axis("off")
    ax_border.add_patch(
        Rectangle((0.005, 0.01), 0.99, 0.98, fill=False, lw=cfg.border_lw, transform=ax_border.transAxes)
    )

    right_w = cfg.right_col_frac
    left_w = 1.0 - right_w

    y0 = 0.01
    y1 = 0.99
    H = y1 - y0
    logo_h = 0.22 * H
    info_h = 0.52 * H
    leg_h = H - logo_h - info_h

    # eixo do mapa
    ax_map = fig.add_axes([0.01, 0.02, left_w - 0.02, 0.96])
    ax_map.set_axis_off()

    # eixo logo
    ax_logo = fig.add_axes([left_w + 0.01, y1 - logo_h, right_w - 0.02, logo_h - 0.01])
    ax_logo.set_axis_off()
    ax_logo.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=cfg.border_lw, transform=ax_logo.transAxes))

    # eixo info
    ax_info = fig.add_axes([left_w + 0.01, y1 - logo_h - info_h, right_w - 0.02, info_h - 0.01])
    ax_info.set_axis_off()
    ax_info.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=cfg.border_lw, transform=ax_info.transAxes))

    # eixo legenda
    ax_leg = fig.add_axes([left_w + 0.01, y0 + 0.01, right_w - 0.02, leg_h - 0.02])
    ax_leg.set_axis_off()
    ax_leg.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=cfg.border_lw, transform=ax_leg.transAxes))

    # divisória vertical
    ax_border.add_line(
        plt.Line2D([left_w, left_w], [y0, y1], lw=cfg.border_lw, color="black", transform=ax_border.transAxes)
    )

    # título do mapa (em coords do eixo)
    ax_map.text(
        0.0, 0.99, cfg.title_text,
        transform=ax_map.transAxes,
        ha="left", va="top",
        fontsize=cfg.title_fontsize, fontweight="bold"
    )

    # ---------------- Plota mapa ----------------
    gdf["_dose_bin"] = bins
    gdf["_color"] = gdf["_dose_bin"].map(color_map)
    mask = gdf["_dose_bin"] < 0
    if mask.any():
        gdf["_color"] = gdf["_color"].astype(object)
        gdf.loc[mask, "_color"] = [(1, 1, 1, 0)] * int(mask.sum())
    gdf.plot(ax=ax_map, color=gdf["_color"], linewidth=0.0)

    if contorno_utm is not None and len(contorno_utm) > 0:
        contorno_utm.boundary.plot(ax=ax_map, color="black", linewidth=1.5)

    # zoom/limites
    try:
        if contorno_utm is not None and len(contorno_utm) > 0:
            minx, miny, maxx, maxy = contorno_utm.total_bounds
        else:
            minx, miny, maxx, maxy = gdf.total_bounds
        pad = 0.03 * max(maxx - minx, maxy - miny)
        ax_map.set_xlim(minx - pad, maxx + pad)
        ax_map.set_ylim(miny - pad, maxy + pad)
    except Exception:
        pass

    _draw_north_arrow(ax_map, cfg)
    _draw_scalebar(ax_map, cfg)

    # ---------------- Logo ----------------
    _draw_logo(ax_logo, logo_path, mode=cfg.logo_mode)

    # ---------------- Info ----------------
    _box_title_fig(fig, ax_info, cfg.info_title, cfg)

    produtor = info.get("produtor", "—")
    fazenda = info.get("fazenda", "—")
    talhao = info.get("talhao", info.get("nome_talhao", "—"))
    safra = info.get("safra", info.get("ano_safra", "—"))
    produto = info.get("produto", info.get("produto_recomendado", "—"))

    linhas = [
        ("Produtor", produtor),
        ("Fazenda", fazenda),
        ("Talhão", talhao),
        ("Ano/Safra", safra),
        ("Produto", produto),
        ("Área total (ha)", _fmt_ptbr(area_total_ha, 2)),
        (f"Quantidade total ({qty_unit})", _fmt_ptbr(total_qty, 2)),
        (f"Taxa-alvo mínima ({unit})", _fmt_ptbr(dose_min, decimals)),
        (f"Taxa-alvo média ({unit})", _fmt_ptbr(dose_mean, decimals)),
        (f"Taxa-alvo máxima ({unit})", _fmt_ptbr(dose_max, decimals)),
    ]

    block_height = 0.083
    line_gap = 0.0345
    
    y = 1.0 - (cfg.header_height_fig / ax_info.get_position().height) - 0.04
    for k, v in linhas:
        ax_info.text(0.05, y, f"{k}:", transform=ax_info.transAxes,
                     ha="left", va="top",
                     fontsize=cfg.info_fontsize, fontweight="bold")
    
        ax_info.text(0.05, y - line_gap, f"{v}", transform=ax_info.transAxes,
                     ha="left", va="top",
                     fontsize=cfg.info_fontsize)
    
        y -= block_height


    # ---------------- Legenda ----------------
    _box_title_fig(fig, ax_leg, cfg.legend_title, cfg)

    handles = []
    labels = []
    for v in classes:
        handles.append(Patch(facecolor=color_map[int(v)], edgecolor="black", linewidth=0.5))
        labels.append(labels_5[int(v)])

    ncol = 1

    ax_leg.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(0.05, 0.4),
        frameon=False,
        ncol=ncol,
        fontsize=cfg.legend_fontsize,
        handlelength=1.5,
        labelspacing=0.6,
        borderaxespad=0.0,
    )

    # ---------------- Save PDF ----------------
    out_dir = os.path.dirname(output_pdf_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig)

    plt.close(fig)
    _log(logger, "INFO", f"Relatório gerado: {output_pdf_path}")
    return output_pdf_path


def gerar_relatorios_em_lote(
    prescricoes: Sequence[Dict[str, Any]],
    output_dir: str,
    logo_path: Optional[str],
    cfg: Optional[RelatorioConfig] = None,
    logger=None,
) -> List[str]:
    """
    Gera vários PDFs (1 por item).
    Cada item deve ter:
      - gdf_prescricao
      - contorno_utm
      - info (dict)
      - nome_arquivo (opcional)
    """
    cfg = cfg or RelatorioConfig()
    os.makedirs(output_dir, exist_ok=True)

    out_paths = []
    for i, item in enumerate(prescricoes, start=1):
        gdf = item["gdf_prescricao"]
        cont = item["contorno_utm"]
        info = item.get("info", {})
        nome = item.get("nome_arquivo") or f"prescricao_{i:02d}.pdf"
        out_pdf = os.path.join(output_dir, nome)

        out_paths.append(
            gerar_relatorio_prescricao(
                gdf_prescricao=gdf,
                contorno_utm=cont,
                output_pdf_path=out_pdf,
                logo_path=logo_path,
                info=info,
                cfg=cfg,
                logger=logger,
            )
        )
    return out_paths
