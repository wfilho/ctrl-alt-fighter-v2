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


class RuiSpritesheetTest(unittest.TestCase):
    def test_non_special_frames_have_one_coherent_body(self):
        source = Path(__file__).parents[1] / "public" / "assets" / "spritesheets-v3" / "rui-spritesheet.png"
        sheet = np.array(Image.open(source).convert("RGBA"))
        self.assertEqual(
            tuple(sheet.shape[:2]),
            (REPACK.ROWS * REPACK.FRAME_H, REPACK.COLS * REPACK.FRAME_W),
            "Rui spritesheet must use the runtime 320x256 cells",
        )

        for row in [0, 1, 2, 3, 4, 5, 7]:
            for col in range(REPACK.COLS):
                cell = sheet[
                    row * REPACK.FRAME_H:(row + 1) * REPACK.FRAME_H,
                    col * REPACK.FRAME_W:(col + 1) * REPACK.FRAME_W,
                ]
                mask = cell[:, :, 3] > REPACK.ALPHA_CUTOFF
                labels, sizes = REPACK.label_components(mask)
                self.assertGreaterEqual(len(sizes), 1, f"Rui frame r{row}c{col} is empty")

                main = int(np.argmax(sizes)) + 1
                main_mask = labels == main
                ys = np.where(main_mask.any(axis=1))[0]
                self.assertGreaterEqual(
                    ys[-1] - ys[0] + 1,
                    160,
                    f"Rui frame r{row}c{col} has no complete body",
                )

                if len(sizes) > 1:
                    second_largest = sorted(sizes, reverse=True)[1]
                    self.assertLess(
                        second_largest / sizes[main - 1],
                        0.35,
                        f"Rui frame r{row}c{col} has a second large vertical body block",
                    )


if __name__ == "__main__":
    unittest.main()
