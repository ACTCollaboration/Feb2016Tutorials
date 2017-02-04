import unittest
import actExamples.likelihood as lhood
import numpy as np
from ConfigParser import SafeConfigParser 

def listFromConfig(Config,section,name):
    return [float(x) for x in Config.get(section,name).split(',')]

class LikeTests(unittest.TestCase):
    
    def setUp(self):
        iniFile = "tests/config.ini"
        self.Config = SafeConfigParser()
        self.Config.optionxform=str
        self.Config.read(iniFile)

    def test_model(self):
        model_true = np.loadtxt(self.Config.get("files","testModelVec"))
        vec = np.asarray(listFromConfig(self.Config,"makeTest","testVec"))
        paramA = self.Config.getfloat("makeTest","paramA")
        paramB = self.Config.getfloat("makeTest","paramB")
        ans = lhood.model(vec,paramA,paramB)
        self.assertEqual(model_true.tolist(), ans.tolist())

    def test_lnLike(self):
        model = np.loadtxt(self.Config.get("files","testModelVec"))
        data = np.loadtxt(self.Config.get("files","testDataVec"))
        invc = np.loadtxt(self.Config.get("files","testInvCov"))
        ans = lhood.logLike(model,data,invc)
        expected = self.Config.getfloat("makeTest","expected")
        self.assertAlmostEqual(ans, expected,places=self.Config.getint("makeTest","precisionPlaces"))
        

if __name__ == '__main__':
    unittest.main()        
