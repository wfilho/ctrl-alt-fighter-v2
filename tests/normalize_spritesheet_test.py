import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "normalize-rui-spritesheet.py"
SPEC = importlib.util.spec_from_file_location("normalize_spritesheet", SCRIPT)
NORMALIZE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(NORMALIZE)


class NormalizeSpritesheetTest(unittest.TestCase):
    def test_normalizes_a_non_divisible_4x8_source(self):
        source = Image.new("RGBA", (1402, 1122), (0, 0, 0, 0))
        pixels = np.array(source)
        for row in range(8):
            for col in range(4):
                x0 = round(col * 1402 / 4)
                x1 = round((col + 1) * 1402 / 4)
                y0 = round(row * 1122 / 8)
                y1 = round((row + 1) * 1122 / 8)
                pixels[y0:y1, x0:x1, 3] = 255
        source = Image.fromarray(pixels)

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            destination_path = Path(directory) / "normalized.png"
            source.save(source_path)
            NORMALIZE.normalize(source_path, destination_path)
            result = Image.open(destination_path).convert("RGBA")

        self.assertEqual(result.size, (1280, 2048))
        alpha = np.array(result)[:, :, 3]
        for row in range(8):
            for col in range(4):
                cell = alpha[row * 256 : (row + 1) * 256, col * 320 : (col + 1) * 320]
                self.assertGreater(cell.max(), 0, f"normalized frame r{row}c{col} is empty")


if __name__ == "__main__":
    unittest.main()
