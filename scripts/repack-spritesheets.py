#!/usr/bin/env python3
"""Reempacota as sprite sheets dos lutadores.

Problemas que este script corrige nas folhas de origem:

1. Fragmentos soltos ("sapato flutuando", pedaco de perna no topo da celula)
   que vieram da arte original e aparecem em jogo como lixo visual.
2. Personagens com alturas diferentes entre si (um parece maior que o outro)
   e alturas inconsistentes entre quadros do MESMO personagem, o que faz a
   animacao "pulsar" de tamanho.
3. Personagem descentralizado / sem alinhamento de pe no chao.

Estrategia:
  - Em cada celula, isola o corpo real por componentes conectados. Mantem o
    maior componente e qualquer pedaco proximo dele (braco/perna separados por
    uma falha de contorno). Descarta fragmentos distantes.
  - Escala TODOS os quadros de um personagem pelo MESMO fator, calculado a
    partir da altura media da linha de idle, de forma que todos os personagens
    fiquem com a mesma altura de idle. Usar um unico fator por personagem
    preserva as diferencas legitimas de pose (agachar e mais baixo que idle)
    sem fazer o tamanho oscilar durante a animacao.
  - Alinha os pes numa linha de base fixa e centraliza horizontalmente pela
    ancora dos pes (parte de baixo do corpo), que e estavel mesmo quando o
    personagem estica o braco ou a perna num golpe.

Uso:
    python3 scripts/repack-spritesheets.py ORIGEM DESTINO
"""

import sys
import os
import glob

import numpy as np
from PIL import Image

FRAME_W, FRAME_H = 320, 256
COLS, ROWS = 4, 8
POSES = ['idle', 'walk', 'punch', 'kick', 'guard', 'jump', 'special', 'hurt']

# Linha do chao dentro da celula e folgas maximas.
BASELINE = 248
MAX_BODY_H = 240
MAX_BODY_W = 308

# Altura alvo do corpo na pose idle, igual para todos os personagens.
TARGET_IDLE_H = 218

# Raio usado para decidir se um pedaco solto ainda faz parte do corpo.
# Medicao nas folhas de origem: membros legitimos separados por falha de
# contorno ficam a 2-30px do corpo; lixo flutuante (o "sapato solto") aparece
# a 43px ou mais. 34px separa os dois grupos com folga.
NEAR_RADIUS = 34
# Area minima (em pixels) para um pedaco solto ser considerado parte do corpo.
MIN_FRAGMENT_AREA = 120
# Um fragmento tambem precisa representar uma parte relevante do corpo. Isso
# elimina cabecas/tenis vazados de outra pose sem remover pernas e bracos que
# ficaram separados por uma falha de transparencia.
MIN_FRAGMENT_RATIO = 0.15
ALPHA_CUTOFF = 24
# Faixas verticais separadas por ate este numero de linhas vazias sao partes
# do mesmo corpo (falha de contorno), nao poses diferentes. Medicao: cortes
# internos de corpo ficam em 3-10px; a proxima pose comeca 20px ou mais longe.
MERGE_BAND_GAP = 11
# Fracao da maior faixa que uma faixa precisa ter para ser considerada uma
# figura (e nao poeira/sombra/faisca) na hora de escolher a pose apoiada.
BODY_BAND_RATIO = 0.35


def label_components(mask):
    """Rotula componentes 4-conectados sem depender de scipy."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    sizes = []
    next_label = 0
    height, width = mask.shape

    for y, x in zip(*np.where(mask)):
        if labels[y, x]:
            continue

        next_label += 1
        labels[y, x] = next_label
        stack = [(int(y), int(x))]
        size = 0
        while stack:
            current_y, current_x = stack.pop()
            size += 1
            for neighbor_y, neighbor_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and mask[neighbor_y, neighbor_x]
                    and not labels[neighbor_y, neighbor_x]
                ):
                    labels[neighbor_y, neighbor_x] = next_label
                    stack.append((neighbor_y, neighbor_x))
        sizes.append(size)

    return labels, np.asarray(sizes, dtype=np.int64)


def is_near_main(component, main_mask, radius):
    """Retorna se algum pixel do componente esta na vizinhanca do corpo."""
    component_rows = np.where(component.any(axis=1))[0]
    for y in component_rows:
        component_xs = np.where(component[y])[0]
        x0 = max(0, int(component_xs.min()) - radius)
        x1 = min(component.shape[1], int(component_xs.max()) + radius + 1)
        y0 = max(0, int(y) - radius)
        y1 = min(component.shape[0], int(y) + radius + 1)
        if main_mask[y0:y1, x0:x1].any():
            return True
    return False


def main_band(mask):
    """Recorta a faixa vertical que contem o personagem.

    O problema medido na arte de origem: muitas celulas contem DUAS figuras --
    a pose correta daquela celula e um pedaco de outra pose que vazou (em
    geral as pernas de pulo). O invasor chega a 15.000px, entao filtrar por
    area nao resolve; e a posicao tambem nao, porque as vezes o invasor esta
    embaixo (Rui) e as vezes em cima (Kalliane).

    O que funciona e agrupar: um corpo vem partido em varias faixas separadas
    por poucas linhas vazias (cintura, cinto, colete), enquanto figuras
    diferentes ficam bem mais afastadas. Agrupando faixas proximas
    (MERGE_BAND_GAP) cada grupo passa a ser uma figura inteira, e entao basta
    ficar com o grupo de maior area. Verificado nas 9 folhas: isso resolve as
    288 celulas sem nenhum caso ambiguo restante.
    """
    filled = mask.any(axis=1)
    bands = []
    start = None
    for y, has_pixel in enumerate(filled):
        if has_pixel and start is None:
            start = y
        elif not has_pixel and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(filled)))

    if not bands:
        return mask

    # Agrupa faixas proximas: cada grupo vira uma figura completa.
    groups = [list(bands[0])]
    for band in bands[1:]:
        if band[0] - groups[-1][1] <= MERGE_BAND_GAP:
            groups[-1][1] = band[1]
        else:
            groups.append(list(band))

    top, bottom = max(groups, key=lambda g: mask[g[0]:g[1]].sum())

    band_mask = np.zeros_like(mask)
    band_mask[top:bottom] = mask[top:bottom]
    return band_mask


def clean_cell(cell_rgba):
    """Remove fragmentos soltos e devolve (rgba_limpo, mascara_do_corpo)."""
    mask = cell_rgba[:, :, 3] > ALPHA_CUTOFF
    if not mask.any():
        return None, None

    mask = main_band(mask)
    if not mask.any():
        return None, None

    labels, sizes = label_components(mask)
    count = len(sizes)
    main = int(np.argmax(sizes)) + 1
    main_mask = labels == main
    main_area = int(sizes[main - 1])

    keep = main_mask.copy()
    for index in range(1, count + 1):
        if index == main:
            continue
        if sizes[index - 1] < max(MIN_FRAGMENT_AREA, main_area * MIN_FRAGMENT_RATIO):
            continue
        component = labels == index

        # main_band ja isolou a figura correta; aqui so reanexamos pedacos do
        # proprio corpo que ficaram soltos por falha de contorno. Um pedaco
        # grande porem separado por uma faixa vazia larga e a outra figura
        # desenhada na mesma celula, entao NAO entra.
        component_ys = np.where(component.any(axis=1))[0]
        main_ys = np.where(main_mask.any(axis=1))[0]
        if component_ys.max() < main_ys.min():
            vertical_gap = int(main_ys.min()) - int(component_ys.max()) - 1
        elif main_ys.max() < component_ys.min():
            vertical_gap = int(component_ys.min()) - int(main_ys.max()) - 1
        else:
            vertical_gap = 0
        if vertical_gap > MERGE_BAND_GAP:
            continue

        if is_near_main(component, main_mask, NEAR_RADIUS):
            keep |= component

    cleaned = cell_rgba.copy()
    cleaned[~keep] = 0
    return cleaned, keep


def body_metrics(keep_mask):
    """Devolve (x0, y0, x1, y1, ancora_x) do corpo."""
    ys, xs = np.where(keep_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # Ancora horizontal: centro da faixa inferior do corpo (os pes). Estavel
    # mesmo quando o personagem estica o braco num soco.
    height = y1 - y0
    foot_band = max(1, int(height * 0.12))
    foot_rows = keep_mask[y1 - foot_band:y1, :]
    foot_xs = np.where(foot_rows.any(axis=0))[0]
    anchor_x = float(foot_xs.mean()) if foot_xs.size else float((x0 + x1) / 2)

    return x0, y0, x1, y1, anchor_x


def analyse(path):
    """Extrai os 32 quadros limpos + metricas de uma folha."""
    sheet = np.array(Image.open(path).convert('RGBA'))
    frames = {}
    for row in range(ROWS):
        for col in range(COLS):
            cell = sheet[row * FRAME_H:(row + 1) * FRAME_H,
                         col * FRAME_W:(col + 1) * FRAME_W]
            cleaned, keep = clean_cell(cell)
            if cleaned is None:
                frames[(row, col)] = None
                continue
            frames[(row, col)] = (cleaned, keep, body_metrics(keep))
    return frames


GUARD_ROW = 4
GROUNDED_TOLERANCE = 18


def fix_guard_row(frames, report, name):
    """Garante que a linha de defesa use apenas quadros com os pes no chao.

    Em 5 das 9 folhas de origem a linha de defesa e byte-a-byte identica a
    linha de pulo, entao o personagem aparecia flutuando/no ar enquanto
    defendia. Como nao da para inventar arte nova, usamos o quadro mais
    "plantado no chao" da propria linha como pose de defesa. Defender e uma
    pose sustentada em jogos de luta, entao um quadro estavel e o correto.
    """
    grounded = []
    for col in range(COLS):
        entry = frames.get((GUARD_ROW, col))
        if not entry:
            continue
        _, keep, (_, _, _, y1, _) = entry
        grounded.append((y1, col))

    if not grounded:
        return

    lowest = max(y1 for y1, _ in grounded)
    floating = [col for y1, col in grounded if lowest - y1 > GROUNDED_TOLERANCE]
    if not floating:
        return

    best_col = max(grounded)[1]
    for col in floating:
        frames[(GUARD_ROW, col)] = frames[(GUARD_ROW, best_col)]
    report.append(
        f"    defesa: {len(floating)} quadro(s) no ar trocados pelo quadro "
        f"apoiado #{best_col}")


def repack(path, out_path, report):
    frames = analyse(path)
    name = os.path.basename(path)
    header_index = len(report)
    report.append(name)
    fix_guard_row(frames, report, name)

    idle_heights = []
    for col in range(COLS):
        entry = frames.get((0, col))
        if entry:
            _, _, (_, y0, _, y1, _) = entry
            idle_heights.append(y1 - y0)
    if not idle_heights:
        report[header_index] = f"{name}: sem linha de idle utilizavel, pulado"
        return

    idle_h = sum(idle_heights) / len(idle_heights)
    scale = TARGET_IDLE_H / idle_h

    # Garante que nenhum quadro estoure a celula depois da escala.
    for entry in frames.values():
        if not entry:
            continue
        _, _, (x0, y0, x1, y1, _) = entry
        scale = min(scale,
                    MAX_BODY_H / (y1 - y0),
                    MAX_BODY_W / (x1 - x0))

    out = np.zeros((FRAME_H * ROWS, FRAME_W * COLS, 4), np.uint8)
    heights = []

    for (row, col), entry in frames.items():
        if not entry:
            continue
        cleaned, keep, (x0, y0, x1, y1, anchor_x) = entry

        crop = Image.fromarray(cleaned[y0:y1, x0:x1])
        new_w = max(1, int(round((x1 - x0) * scale)))
        new_h = max(1, int(round((y1 - y0) * scale)))
        crop = crop.resize((new_w, new_h), Image.LANCZOS)
        heights.append(new_h)

        # Ancora dos pes vai para o centro da celula; pes na linha de base.
        anchor_in_crop = (anchor_x - x0) * scale
        dest_x = int(round(FRAME_W / 2 - anchor_in_crop))
        dest_y = BASELINE - new_h

        dest_x = max(0, min(dest_x, FRAME_W - new_w))
        dest_y = max(0, min(dest_y, FRAME_H - new_h))

        patch = np.array(crop)
        target = out[row * FRAME_H + dest_y:row * FRAME_H + dest_y + new_h,
                     col * FRAME_W + dest_x:col * FRAME_W + dest_x + new_w]
        target[...] = patch

    Image.fromarray(out).save(out_path)
    report[header_index] = (
        f"{name}: escala={scale:.3f} idle_origem={idle_h:.0f}px "
        f"alturas_finais={min(heights)}-{max(heights)}px")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'public/assets/spritesheets-v2'
    dst = sys.argv[2] if len(sys.argv) > 2 else 'public/assets/spritesheets-v3'
    os.makedirs(dst, exist_ok=True)

    report = []
    for path in sorted(glob.glob(os.path.join(src, '*-spritesheet.png'))):
        repack(path, os.path.join(dst, os.path.basename(path)), report)

    print('\n'.join(report))


if __name__ == '__main__':
    main()
