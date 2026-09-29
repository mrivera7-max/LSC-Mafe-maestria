"""
Generación de informes HTML autocontenidos (sin dependencias externas) + CSV.

Reglas de visualización: un color fijo por modelo (comparacion.COLORES),
un solo eje por gráfica, valores siempre escritos en texto (nunca solo color),
matrices de confusión en escala secuencial de un solo tono.
"""

import csv
import html
from pathlib import Path

from comparacion import COLORES, NOMBRES_LEGIBLES

CSS = """
:root{--bg:#fbfbf9;--panel:#ffffff;--txt:#1d1d1b;--txt2:#5f5e58;--linea:#e4e3de;
--acento:#0f3460;--ok:#008300;--mal:#c8321e}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--panel:#232322;--txt:#f4f4f1;
--txt2:#c3c2b7;--linea:#3a3a37;--acento:#7fb0ef}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 "Segoe UI",system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid var(--linea);padding-bottom:4px}
h3{font-size:15px;margin:20px 0 6px}
.sub{color:var(--txt2);margin:0 0 16px}
.nota{color:var(--txt2);font-size:13px}
.aviso{border-left:3px solid var(--mal);padding:8px 12px;background:var(--panel);margin:12px 0}
table{border-collapse:collapse;width:100%;margin:8px 0;background:var(--panel);font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--linea);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--txt2);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.scroll{overflow-x:auto}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:12px 0}
.tile{background:var(--panel);border:1px solid var(--linea);border-radius:8px;padding:12px}
.tile .k{color:var(--txt2);font-size:12px}.tile .v{font-size:22px;font-weight:600}
.graf{background:var(--panel);border:1px solid var(--linea);border-radius:8px;padding:12px 16px;margin:8px 0}
.fila{display:grid;grid-template-columns:150px 1fr 110px;align-items:center;gap:10px;margin:6px 0}
.pista{position:relative;height:18px}
.barra{position:absolute;left:0;top:0;height:18px;border-radius:0 4px 4px 0}
.err{position:absolute;top:8px;height:2px;background:var(--txt)}
.err::before,.err::after{content:"";position:absolute;top:-5px;width:2px;height:12px;background:var(--txt)}
.err::before{left:0}.err::after{right:0}
.val{text-align:right;font-variant-numeric:tabular-nums}
.cm td{text-align:center;min-width:44px}
.cm td.et{text-align:left;color:var(--txt2)}
.cms{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:16px}
details{margin:8px 0}summary{cursor:pointer;color:var(--acento)}
"""

# Rampa secuencial azul (claro -> oscuro) para matrices de confusión
RAMPA = ["#f5f9fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def esc(x) -> str:
    return html.escape(str(x))


def nombre_modelo(m: str) -> str:
    return (f'<span class="sw" style="background:{COLORES.get(m, "#888")}"></span>'
            f'{esc(NOMBRES_LEGIBLES.get(m, m))}')


def tabla(columnas, filas, clase="") -> str:
    th = "".join(f"<th>{c}</th>" for c in columnas)
    tr = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in f) + "</tr>" for f in filas)
    return f'<div class="scroll"><table class="{clase}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def barras(titulo, items, unidad="", maximo=None, decimales=1, nota="") -> str:
    """items: [(modelo, valor, error_o_None, texto_tooltip)] — barra horizontal, un eje."""
    if not items:
        return ""
    tope = maximo or max((v + (e or 0)) for _, v, e, _ in items) or 1.0
    filas = []
    for m, v, e, tip in items:
        ancho = max(0.0, min(100.0, 100 * v / tope))
        err = ""
        if e:
            a = max(0.0, 100 * (v - e) / tope)
            b = min(100.0, 100 * (v + e) / tope)
            err = f'<div class="err" style="left:{a:.2f}%;width:{b - a:.2f}%"></div>'
        txt = f"{v:.{decimales}f}{unidad}" + (f" ± {e:.{decimales}f}" if e else "")
        filas.append(
            f'<div class="fila" title="{esc(tip or txt)}"><div>{nombre_modelo(m)}</div>'
            f'<div class="pista"><div class="barra" style="width:{ancho:.2f}%;background:{COLORES.get(m, "#888")}"></div>{err}</div>'
            f'<div class="val">{txt}</div></div>')
    n = f'<p class="nota">{nota}</p>' if nota else ""
    return f'<div class="graf"><h3>{esc(titulo)}</h3>{"".join(filas)}{n}</div>'


def matriz_confusion(cm, clases, titulo, normalizar=True) -> str:
    import numpy as np
    cm = np.asarray(cm)
    filas_sum = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, filas_sum, out=np.zeros_like(cm, dtype=float), where=filas_sum > 0)
    cab = "<tr><th>real \\ pred.</th>" + "".join(f"<th>{esc(c)}</th>" for c in clases) + "</tr>"
    cuerpo = []
    for i, c in enumerate(clases):
        celdas = [f'<td class="et">{esc(c)}</td>']
        for j in range(len(clases)):
            f = frac[i, j]
            color = RAMPA[min(len(RAMPA) - 1, int(round(f * (len(RAMPA) - 1))))]
            tinta = "#ffffff" if f >= 0.55 else "#1d1d1b"
            tip = f"real {c} → pred. {clases[j]}: {cm[i, j]} ({f*100:.0f}%)"
            celdas.append(f'<td style="background:{color};color:{tinta}" title="{esc(tip)}">{cm[i, j]}</td>')
        cuerpo.append("<tr>" + "".join(celdas) + "</tr>")
    return (f'<div class="graf"><h3>{titulo}</h3><div class="scroll"><table class="cm">'
            f'<thead>{cab}</thead><tbody>{"".join(cuerpo)}</tbody></table></div>'
            f'<p class="nota">Conteos; color = fracción de la fila (recall por clase).</p></div>')


def tiles(pares) -> str:
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in pares
    ) + "</div>"


def pagina(titulo, subtitulo, cuerpo) -> str:
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(titulo)}</title><style>{CSS}</style></head><body><main>'
            f'<h1>{esc(titulo)}</h1><p class="sub">{subtitulo}</p>{cuerpo}</main></body></html>')


def escribir_csv(ruta: Path, columnas, filas):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel abre bien las tildes
        w = csv.writer(f, delimiter=";")
        w.writerow(columnas)
        w.writerows(filas)


def entorno_html(ent: dict) -> str:
    filas = [[k, esc(v)] for k, v in ent.items() if k != "paquetes"]
    filas += [[f"paquete: {k}", esc(v)] for k, v in ent.get("paquetes", {}).items()]
    return tabla(["Elemento", "Valor"], filas)
