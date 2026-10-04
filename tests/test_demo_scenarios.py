"""Chemical interpretation and reactive scenario contracts, without expensive fitting."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from genotox_food_migrants.demo_engine import build_demo
from genotox_food_migrants.demo_view import build_demo_view
from genotox_food_migrants import models, splitting, chemistry

class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.demo = build_demo(Path(__file__))
        self.features = pd.DataFrame([dict(feature_id='fixture', msi_level=2,
            n_isomers_total=2484, sampled=True, chemical_formula='C2H6O',
            display_name='Assigned fixture', assigned_uri='https://example.test', score=.05,
            tanimoto_neighbour=.8)])
        self.scored = pd.DataFrame([
            dict(feature_id='fixture', rank=i+1, cid=i+1, is_assigned=i==0,
                 inchikey=f'raw-{i}', inchikey_std=f'parent-{i}', smiles='CCO', smiles_std='CCO',
                 score=score, raw_score=score, tanimoto_neighbour=similarity,
                 in_training=i==1, nn_smiles='CCO', nn_name='neighbour', nn_inchikey='nn', structure_note='')
            for i,(score,similarity) in enumerate([(.05,.8),(.25,.6),(np.nan,np.nan)])])
    def evaluate(self, cutoff=.4, assumption='same_formula'):
        return self.demo.evaluate(self.scored, self.features, 'fixture', cutoff, assumption)

    def test_zero_denominator_is_unavailable_in_view_and_json(self):
        view=self.evaluate(.9)
        self.assertEqual(view['covered_weight'], 0)
        self.assertIsNone(view['conditional_score'])
        exported=json.loads(self.demo.snapshot(view, {}))
        self.assertIsNone(exported['scenario_average_over_retained_structures'])
        self.assertNotIn('NaN', json.dumps(exported))
        html=build_demo_view().scenario_summary(view)
        self.assertIn('Unavailable', html)
        self.assertIn('no average exists', html)

    def test_only_included_weight_enters_average(self):
        view=self.evaluate()
        self.assertAlmostEqual(view['conditional_score'], .15)
        self.assertAlmostEqual(view['covered_weight'], 2/3)
        self.assertAlmostEqual(view['retained_pool_fraction'], 3/2484)
        self.assertEqual(view['n_evaluable'],2)
        self.assertEqual(view['listed_formula_pool_size'],2484)

    def test_sampling_and_coverage_descriptions_are_separate(self):
        html=build_demo_view().scenario_summary(self.evaluate())
        self.assertIn('2 / 3 retained structures',html)
        self.assertIn('3 / 2,484 listed formula structures',html)
        self.assertIn('hypothetical',html.lower())
        self.assertIn('not the probability that the GC–MS peak is genotoxic',html)

    def test_confirmed_identity_keeps_its_weight(self):
        self.features['msi_level']=1
        view=self.evaluate()
        self.assertAlmostEqual(view['conditional_score'],.05)
        self.assertEqual(view['covered_weight'],1)
        self.assertEqual(view['candidates'].weight.tolist(),[1,0,0])
        self.assertIn('Standard confirmation is preserved',build_demo_view().scenario_summary(view))

    def test_inspection_is_not_scenario_reweighting(self):
        view=self.evaluate()
        inspected=self.demo.inspect_structure(view,'fixture:2')
        self.assertEqual(inspected['selected']['score'],.25)
        self.assertAlmostEqual(view['conditional_score'],.15)
        assigned=self.demo.inspect_structure(self.evaluate(assumption='published'),'fixture:2')
        self.assertTrue(assigned['selected']['is_assigned'])
        other=self.demo.inspect_structure(view,'other:2')
        self.assertTrue(other['selected']['is_assigned'])

    def test_training_identity_and_json_provenance_are_explicit(self):
        view=self.evaluate()
        inspected=self.demo.inspect_structure(view,'fixture:2')
        exported=json.loads(self.demo.snapshot(view,{'model_fit_token':'fixed'},inspected))
        self.assertTrue(exported['inspected_structure']['in_training'])
        self.assertIn('not a held-out',exported['inspected_structure']['prediction_scope'])
        self.assertEqual(exported['provenance']['model_fit_token'],'fixed')
        self.assertIn('TRAINING IDENTITY',build_demo_view().inspector(view,inspected))

    def test_controls_do_not_fit_and_do_not_change_structure_scores(self):
        before=self.scored.score.copy()
        with patch.object(models,'fit_calibrated',side_effect=AssertionError('refit')), \
             patch.object(models,'build',side_effect=AssertionError('refit')):
            for cutoff in [.4,.7,.9]:
                for assumption in ['published','same_formula']:
                    self.evaluate(cutoff,assumption)
        pd.testing.assert_series_equal(before,self.scored.score)

class GroupingTests(unittest.TestCase):
    def test_acyclic_related_structures_stay_in_one_component(self):
        smiles=['CCCCCCCCCCCCCCCCCCCC','CCCCCCCCCCCCCCCCCCCCC','c1ccccc1']
        scaffolds=[chemistry.scaffold(s) for s in smiles]
        fps=chemistry.fingerprints(smiles)
        groups=splitting.identity_groups(scaffolds,['a','b','c'],fps,.70)
        self.assertEqual(groups[0],groups[1])
        self.assertNotEqual(groups[0],groups[2])
        for seed,tr,te in splitting.multiseed_splits(groups,range(5),.5):
            self.assertFalse(set(groups[tr]) & set(groups[te]))

    def test_stable_identity_groups_survive_reindexing(self):
        a=splitting.identity_groups(['','ring',''],['first','middle','last'])
        b=splitting.identity_groups(['',''],['first','last'])
        self.assertEqual(a[[0,2]].tolist(),b.tolist())

if __name__=='__main__':
    unittest.main()
