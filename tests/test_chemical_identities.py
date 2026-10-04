"""Regression checks for the chemical/label failures observed in the EFSA data."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from genotox_food_migrants import chemistry, data, external, labels, ttc


class ChemicalIdentityTests(unittest.TestCase):
    def test_metal_sulfates_do_not_become_sulfuric_acid(self):
        for smi in ("[O-]S(=O)(=O)[O-].[Cu+2]",
                    "O.O.O.O.O.O.O.[O-]S(=O)(=O)[O-].[Co+2]"):
            row = chemistry.standardize_mol(smi)
            self.assertFalse(row["ok"])
            self.assertIsNone(row["inchikey_std"])
            self.assertTrue(row["note"].startswith("outside model scope"))

    def test_multicomponent_name_resolution_is_not_acrylamide(self):
        bad = chemistry.standardize_mol("CC(=O)C.CC(=O)C.C=CC(=O)N")
        self.assertFalse(bad["ok"])
        self.assertIn("multiple carbon-containing fragments", bad["note"])
        self.assertTrue(chemistry.standardize_mol("C=CC(=O)N")["ok"])

    def test_supported_counterion_has_the_same_organic_parent(self):
        salt = chemistry.standardize_mol("CC(=O)[O-].[Na+]")
        acid = chemistry.standardize_mol("CC(=O)O")
        self.assertTrue(salt["ok"])
        self.assertEqual(salt["inchikey_std"], acid["inchikey_std"])

    def _records(self):
        # 5 positive conclusions for one raw identity and 3 negative for
        # another parent-equivalent raw identity. A vote over binary labels
        # would tie; the actual conclusion-count majority is positive.
        names = ["acid"] * 5 + ["sodium salt"] * 3
        smiles = ["CC(=O)O"] * 5 + ["CC(=O)[O-].[Na+]"] * 3
        raw = ["acid-key"] * 5 + ["salt-key"] * 3
        return pd.DataFrame({"CleanSubstanceName": names, "smiles": smiles,
                             "inchikey": raw,
                             "Genotoxicity": ["Positive"] * 5 + ["Negative"] * 3})

    def test_rules_use_conclusion_counts_after_standardization(self):
        for rule, expected in (("any_positive", 1), ("majority", 1), ("consensus", None)):
            agg = labels.aggregate_by_substance(self._records(), rule=rule)
            std = chemistry.standardize_table(agg)
            forward = labels.aggregate_standardized(std, rule=rule)
            reverse = labels.aggregate_standardized(std.iloc[::-1], rule=rule)
            pd.testing.assert_frame_equal(forward, reverse)
            self.assertEqual(len(forward), 1)
            self.assertEqual(forward.n_pos.iloc[0], 5)
            self.assertEqual(forward.n_neg.iloc[0], 3)
            if expected is None:
                self.assertTrue(pd.isna(forward.y.iloc[0]))
            else:
                self.assertEqual(forward.y.iloc[0], expected)

    def test_pre_fix_cache_cannot_reintroduce_a_metal_parent(self):
        smi = "[O-]S(=O)(=O)[O-].[Cu+2]"
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "old.parquet"
            pd.DataFrame({"smiles": [smi], "smiles_std": ["OS(=O)(=O)O"],
                          "inchikey_std": ["wrong-parent"], "ok": [True]}).to_parquet(cache)
            out = chemistry.standardize_table(pd.DataFrame({"smiles": [smi]}), cache_path=cache)
            self.assertFalse(out.ok.iloc[0])
            self.assertTrue(pd.isna(out.inchikey_std.iloc[0]))

    def test_all_original_conflicting_groups_have_supported_inputs_only(self):
        records = data.with_structure_only(data.load_efsa())
        agg = labels.aggregate_by_substance(records)
        std = chemistry.standardize_table(agg)
        reconciled = labels.aggregate_standardized(std)
        self.assertFalse(reconciled.inchikey_std.duplicated().any())
        bad_names = {"Copper sulfate", "Cobalt sulphate heptahydrate",
                     "4-acrylamido-4-methyl-2-pentanone"}
        self.assertFalse(reconciled.name.isin(bad_names).any())
        # The remaining parent-identity labels obey the declared conservative rule.
        informative = reconciled[reconciled.y.notna()]
        self.assertTrue((informative.y.astype(int) == (informative.n_pos > 0).astype(int)).all())

    def test_external_structures_retain_model_eligibility(self):
        base = chemistry.standardize_table(pd.DataFrame({"smiles": ["CCO"], "y": [0]}))
        ext = chemistry.standardize_table(pd.DataFrame({"smiles": ["CCN"], "y": [1], "source": ["test"]}))
        merged, conflicts = external.merge(base, ext)
        self.assertEqual(len(merged), 2)
        self.assertTrue(merged.ok.all())
        self.assertTrue(conflicts.empty)

    def test_external_only_disagreement_is_not_arbitrated_by_row_order(self):
        base = chemistry.standardize_table(pd.DataFrame({"smiles": ["CCO"], "y": [0]}))
        ext = chemistry.standardize_table(pd.DataFrame({"smiles": ["CCN", "CCN"], "y": [1, 0], "source": ["a", "b"]}))
        merged, conflicts = external.merge(base, ext)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(conflicts), 2)

    def test_zero_coverage_does_not_produce_a_ttc_decision(self):
        f = pd.DataFrame({"feature_id": ["uncovered", "covered"],
                          "identity_weighted_genotoxicity": [0., 0.],
                          "covered_weight": [0., 1.]})
        base = pd.DataFrame({"feature_id": ["uncovered", "covered"], "ttc_baseline": [90., 90.]})
        out = ttc.decide(f, base)
        self.assertTrue(pd.isna(out.ttc_model.iloc[0]))
        self.assertTrue(pd.isna(out.calls_genotoxic.iloc[0]))
        self.assertEqual(out.ttc_model.iloc[1], 90.)


if __name__ == "__main__":
    unittest.main()
