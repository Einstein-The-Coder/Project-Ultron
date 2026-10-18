import unittest

import numpy as np

from ai.preprocessing import prepare_input


class PrepareInputTests(unittest.TestCase):
    def test_reshapes_64_pixels(self):
        result = prepare_input([0] * 64)
        self.assertEqual(result.shape, (1, 64))
        self.assertEqual(result.dtype, np.float32)

    def test_rejects_wrong_number_of_pixels(self):
        with self.assertRaises(ValueError):
            prepare_input([0] * 63)

    def test_rejects_non_finite_values(self):
        for invalid in (float("nan"), float("inf"), float("-inf")):
            values = [0] * 64
            values[0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prepare_input(values)

    def test_rejects_pixels_outside_expected_range(self):
        for invalid in (-0.1, 16.1):
            values = [0] * 64
            values[0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prepare_input(values)


if __name__ == "__main__":
    unittest.main()
