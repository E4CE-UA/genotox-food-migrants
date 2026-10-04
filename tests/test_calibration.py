"""Calibration must keep chemical groups and fitted targets independent."""
import unittest
import numpy as np
from genotox_food_migrants import models


class CalibrationTests(unittest.TestCase):
    def test_each_identity_calibrates_once_without_group_leakage(self):
        scaffolds = np.repeat([f"ring-{i}" for i in range(12)], 4)
        y = np.tile([0, 0, 0, 1], 12)
        cv = models.calibration_splits(y, scaffolds, seed=0)
        seen = np.zeros(len(y), dtype=int)
        for train, calibration in cv:
            self.assertFalse(set(train) & set(calibration))
            self.assertFalse(set(scaffolds[train]) & set(scaffolds[calibration]))
            self.assertEqual(len(np.unique(y[train])), 2)
            self.assertEqual(len(np.unique(y[calibration])), 2)
            seen[calibration] += 1
        np.testing.assert_array_equal(seen, np.ones(len(y), dtype=int))

    def test_one_label_cannot_be_presented_as_calibrated(self):
        with self.assertRaises(ValueError):
            models.calibration_splits(np.zeros(12), [f"g{i}" for i in range(12)])


if __name__ == "__main__":
    unittest.main()
