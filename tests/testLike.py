import unittest
import actExamples.likelihood as lhood
import numpy as np

class LikeTests(unittest.TestCase):


    def test_model(self):
        model_true = np.loadtxt("data/testTheory.txt")
        vec = np.linspace(10.,20.,5)
        ans = lhood.model(vec,3,4)
        self.assertEqual(model_true.tolist(), ans.tolist())

    def test_lnLike(self):
        model = np.loadtxt("data/testTheory.txt")
        data = np.loadtxt("data/testData.txt")
        vec = np.linspace(10.,20.,5)
        ans = lhood.model(vec,3,4)
        self.assertEqual(model_true.tolist(), ans.tolist())
        

if __name__ == '__main__':
    unittest.main()        
