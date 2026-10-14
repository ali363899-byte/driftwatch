import unittest
import math
from monitor import classification,drift,report
from demo import make_data

class MonitoringTests(unittest.TestCase):
    def test_perfect_classifier(self):
        r=classification([{'label':0,'score':0},{'label':1,'score':1}])
        self.assertEqual((r['accuracy'],r['f1'],r['brier'],r['ece']),(1,1,0,0))
    def test_confusion(self):
        rows=[{'label':y,'score':p} for y,p in [(1,.9),(0,.8),(1,.1),(0,.2)]]
        self.assertEqual(classification(rows)['confusion'],{'tp':1,'tn':1,'fp':1,'fn':1})
    def test_undefined_precision(self):
        self.assertIsNone(classification([{'label':0,'score':0}])['precision'])
    def test_threshold_changes_decisions(self):
        self.assertEqual(classification([{'label':1,'score':.7}],.8)['accuracy'],0)
    def test_identical_distributions(self):
        self.assertAlmostEqual(drift(list(range(100)),list(range(100)))['psi'],0)
    def test_shift_increases_psi(self):
        self.assertGreater(drift(list(range(100)),list(range(100,200)))['psi'],.2)
    def test_constant_feature_shift(self):
        self.assertGreater(drift([0]*50,[1]*50)['psi'],.2)
    def test_finite_validation(self):
        for invalid in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError): drift([0,1],[invalid])
    def test_probability_validation(self):
        with self.assertRaises(ValueError): classification([{'label':1,'score':2}])
    def test_mismatched_features(self):
        a,b=make_data(10);b[0]['features']['extra']=1
        with self.assertRaises(ValueError):report(a,b)
    def test_small_segments_flagged(self):
        a,b=make_data(10)
        self.assertTrue(all(v['low_sample'] for v in report(a,b)['segments'].values()))
    def test_reproducible_demo(self):
        self.assertEqual(make_data(5),make_data(5))
    def test_brier_reference(self):
        self.assertAlmostEqual(classification([{'label':1,'score':.5}])['brier'],.25)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):classification([])

if __name__=='__main__':unittest.main()
