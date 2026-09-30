#!/usr/bin/env python3
"""Normaliza uma spritesheet 4x8 do Rui para o contrato do jogo.

O gerador de arte devolve folhas em resolucoes diferentes. Este passo mantem
cada pose dentro da propria celula, preserva a proporcao e alinha a base do
quadro ao mesmo chao usado pelo runtime.
"""

import sys
from pathlib import Path

from PIL import Image


COLS, ROWS = 4, 8
FRAME_W, FRAME_H = 320, 256


def normalize(source: Path, destination: Path) -> None:
    sheet = Image.open(source).convert("RGBA")
    source_w, source_h = sheet.size
    if source_w % COLS or source_h % ROWS:
        raise ValueError(f"A folha {source} nao e divisivel em uma grade 4x8: {sheet.size}")

    source_cell_w = source_w // COLS
    source_cell_h = source_h // ROWS
    result = Image.new("RGBA", (COLS * FRAME_W, ROWS * FRAME_H), (0, 0, 0, 0))

    for row in range(ROWS):
        for col in range(COLS):
            cell = sheet.crop(
                (
                    col * source_cell_w,
                    row * source_cell_h,
                    (col + 1) * source_cell_w,
                    (row + 1) * source_cell_h,
                )
            )

            scale = min(FRAME_W / source_cell_w, FRAME_H / source_cell_h)
            resized = cell.resize(
                (round(source_cell_w * scale), round(source_cell_h * scale)),
                Image.Resampling.LANCZOS,
            )
            x = col * FRAME_W + (FRAME_W - resized.width) // 2
            y = row * FRAME_H + FRAME_H - resized.height
            result.alpha_composite(resized, (x, y))

    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination, "PNG", optimize=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Uso: normalize-rui-spritesheet.py ORIGEM DESTINO")
    normalize(Path(sys.argv[1]), Path(sys.argv[2]))
