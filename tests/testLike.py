import unittest
import actExamples.likelihood as lhood
import numpy as np

class LikeTests(unittest.TestCase):


    def test_model(self):
        model_true = np.loadtxt("tests/data/testTheory.txt")
        vec = np.linspace(10.,20.,5)
        ans = lhood.model(vec,3,4)
        self.assertEqual(model_true.tolist(), ans.tolist())

    def test_lnLike(self):
        model = np.loadtxt("tests/data/testTheory.txt")
        data = np.loadtxt("tests/data/testData.txt")
        invc = np.loadtxt("tests/data/testInv.txt")
        vec = np.linspace(10.,20.,5)
        ans = lhood.logLike(model,data,invc)
        expected = -3145074.28075
        self.assertAlmostEqual(ans, expected,places=5)
        

if __name__ == '__main__':
    unittest.main()        
