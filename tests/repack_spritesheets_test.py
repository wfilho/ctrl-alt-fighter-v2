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


class RepackSpritesheetsTest(unittest.TestCase):
    def test_discards_small_fragment_but_keeps_second_body_segment(self):
        cell = np.zeros((REPACK.FRAME_H, REPACK.FRAME_W, 4), dtype=np.uint8)
        cell[20:100, 110:190, 3] = 255
        cell[111:200, 110:190, 3] = 255
        cell[211:240, 150:180, 3] = 255

        _, keep = REPACK.clean_cell(cell)

        self.assertTrue(keep[20:100, 110:190].any())
        self.assertTrue(keep[111:200, 110:190].any())
        self.assertFalse(keep[211:240, 150:180].any())

    def test_keeps_rui_jump_body_below_the_horizontal_artifact(self):
        source = Path(__file__).parents[1] / "public" / "assets" / "spritesheets-v2" / "rui-spritesheet.png"
        sheet = np.array(Image.open(source).convert("RGBA"))
        cell = sheet[5 * REPACK.FRAME_H:6 * REPACK.FRAME_H, 3 * REPACK.FRAME_W:4 * REPACK.FRAME_W]

        _, keep = REPACK.clean_cell(cell)

        self.assertTrue(keep[180:240, :].any())


if __name__ == "__main__":
    unittest.main()
