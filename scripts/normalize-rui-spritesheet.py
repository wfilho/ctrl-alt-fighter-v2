#!/usr/bin/env python3
"""Normaliza uma spritesheet 4x8 para o contrato do jogo.

O gerador de arte devolve folhas em resolucoes diferentes. Este passo mantem
cada pose dentro da propria celula, preserva a proporcao e alinha a base do
quadro ao mesmo chao usado pelo runtime.
"""

import sys
from pathlib import Path

from PIL import Image


COLS, ROWS = 4, 8
FRAME_W, FRAME_H = 320, 256
CONTENT_MARGIN_X = 8
CONTENT_MARGIN_TOP = 8
BASELINE = 248


def normalize(source: Path, destination: Path) -> None:
    sheet = Image.open(source).convert("RGBA")
    source_w, source_h = sheet.size
    if source_w < COLS or source_h < ROWS:
        raise ValueError(f"A folha {source} e menor que a grade 4x8: {sheet.size}")
    result = Image.new("RGBA", (COLS * FRAME_W, ROWS * FRAME_H), (0, 0, 0, 0))

    for row in range(ROWS):
        for col in range(COLS):
            x0 = round(col * source_w / COLS)
            x1 = round((col + 1) * source_w / COLS)
            y0 = round(row * source_h / ROWS)
            y1 = round((row + 1) * source_h / ROWS)
            cell = sheet.crop((x0, y0, x1, y1))

            alpha = cell.getchannel("A")
            bbox = alpha.getbbox()
            if bbox is None:
                continue
            content = cell.crop(bbox)
            max_width = FRAME_W - 2 * CONTENT_MARGIN_X
            max_height = BASELINE - CONTENT_MARGIN_TOP
            scale = min(max_width / content.width, max_height / content.height)
            resized = content.resize(
                (max(1, round(content.width * scale)), max(1, round(content.height * scale))),
                Image.Resampling.LANCZOS,
            )
            x = col * FRAME_W + (FRAME_W - resized.width) // 2
            y = row * FRAME_H + BASELINE - resized.height
            result.alpha_composite(resized, (x, y))

    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, "PNG", optimize=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Uso: normalize-rui-spritesheet.py ORIGEM DESTINO")
    normalize(Path(sys.argv[1]), Path(sys.argv[2]))
