import unittest

class LikeTests(unittest.TestCase):

    def setUp(self):
        self.testModelVec = np.loadtxt("data/testTheory.txt")
        self.testDataVec = np.loadtxt("data/testData.txt")
        self.testInvCov = np.loadtxt("data/testInv.txt")

    def tearDown(self):
        del self.testModelVec
        del self.testDataVec
        del self.testInvCov

    def test_model(self):
        self.assertEqual(self.testModelVec, range(1, 10))


if __name__ == '__main__':
    unittest.main()        
