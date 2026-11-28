import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dual_camera import LEG_AGREEMENT_TOLERANCE, combine_legs, select_leg_indices  # noqa: E402


class ShankLegTests(unittest.TestCase):
    def test_tolerance_is_15_percent_for_testing(self):
        self.assertEqual(LEG_AGREEMENT_TOLERANCE, 0.15)

    def test_single_leg_is_used_as_is(self):
        width, length, info = combine_legs([{"width": 1.0, "length": 9.0}])
        self.assertEqual((width, length, info["method"], info["disagree"]), (1.0, 9.0, "single", False))

    def test_agreeing_legs_are_averaged(self):
        width, length, info = combine_legs([{"width": 1.0, "length": 10.0}, {"width": 0.9, "length": 9.0}])
        self.assertAlmostEqual(width, 0.95)
        self.assertAlmostEqual(length, 9.5)
        self.assertEqual(info["method"], "mean")
        self.assertFalse(info["disagree"])

    def test_disagreeing_legs_use_largest_mask_leg_and_flag(self):
        # length differs 20% (> 15%) → the first (largest-mask) leg, flagged for review
        width, length, info = combine_legs([{"width": 1.0, "length": 10.0}, {"width": 1.0, "length": 8.0}])
        self.assertEqual((width, length), (1.0, 10.0))
        self.assertEqual(info["method"], "largest")
        self.assertTrue(info["disagree"])
        self.assertAlmostEqual(info["difference"]["length"], 0.2)

    def test_second_leg_selection(self):
        areas, confidences = [1000, 900, 100, 950], [0.9, 0.85, 0.9, 0.5]
        # index 3 is not confident enough, index 2 is a fragment → legs 0 and 1
        self.assertEqual(select_leg_indices(areas, confidences), [0, 1])
        # the same leg detected twice (overlapping) is skipped
        self.assertEqual(select_leg_indices(areas, confidences, overlaps=lambda i, j: 0.9), [0])
        self.assertEqual(select_leg_indices([1000], [0.9]), [0])


if __name__ == "__main__":
    unittest.main()
