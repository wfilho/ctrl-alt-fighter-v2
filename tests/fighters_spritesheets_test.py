import importlib.util
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "repack-spritesheets.py"
SPEC = importlib.util.spec_from_file_location("repack_spritesheets", SCRIPT)
REPACK = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(REPACK)


FIGHTERS = (
    "kalliane",
    "laura",
    "caio",
    "monteiro",
    "vinicius",
    "homologacao",
    "prazo",
    "cliente",
)


class FighterSpritesheetsTest(unittest.TestCase):
    def test_each_animation_frame_contains_one_complete_body(self):
        root = Path(__file__).parents[1] / "public" / "assets" / "spritesheets-v3"

        for fighter in FIGHTERS:
            with self.subTest(fighter=fighter):
                sheet = np.array(
                    Image.open(root / f"{fighter}-spritesheet.png").convert("RGBA")
                )
                self.assertEqual(
                    tuple(sheet.shape[:2]),
                    (REPACK.ROWS * REPACK.FRAME_H, REPACK.COLS * REPACK.FRAME_W),
                )

                for row in [0, 1, 2, 3, 4, 5, 7]:
                    for col in range(REPACK.COLS):
                        cell = sheet[
                            row * REPACK.FRAME_H : (row + 1) * REPACK.FRAME_H,
                            col * REPACK.FRAME_W : (col + 1) * REPACK.FRAME_W,
                        ]
                        mask = cell[:, :, 3] > REPACK.ALPHA_CUTOFF
                        ys, xs = np.where(mask)
                        self.assertTrue(mask.any(), f"{fighter} frame r{row}c{col} is empty")
                        self.assertGreaterEqual(
                            ys[-1] - ys[0] + 1,
                            60,
                            f"{fighter} frame r{row}c{col} has no visible body",
                        )
                        self.assertGreaterEqual(
                            xs.min(),
                            8,
                            f"{fighter} frame r{row}c{col} touches the left cell edge",
                        )
                        self.assertLess(
                            xs.max(),
                            REPACK.FRAME_W - 8,
                            f"{fighter} frame r{row}c{col} touches the right cell edge",
                        )
                        self.assertGreaterEqual(
                            ys.min(),
                            8,
                            f"{fighter} frame r{row}c{col} touches the top cell edge",
                        )
                        self.assertLess(
                            ys.max(),
                            REPACK.FRAME_H - 8,
                            f"{fighter} frame r{row}c{col} touches the bottom cell edge",
                        )


if __name__ == "__main__":
    unittest.main()
