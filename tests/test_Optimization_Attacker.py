import unittest

from optimization_attack.OptimizationAttacker import OptimizationAttacker
import numpy as np

class TestHashNorm(unittest.TestCase):
    def test_output_size(self):
        h1 = np.zeros([5,2])
        h2 = np.zeros([5,2])
        res = OptimizationAttacker.hash_norm(h1,h2)

        self.assertEqual(res.shape,())
    
    def test_different_size_inputs(self):
        h1 = np.zeros([5,2])
        h2 = np.zeros([6,2])
        with self.assertRaises(ValueError):
            res = OptimizationAttacker.hash_norm(h1,h2)
            

    def test_simple_distances(self):
        h1 = np.zeros([3,2])
        h2 = np.zeros([3,2])
        res = OptimizationAttacker.hash_norm(h1,h2)
        self.assertEqual(res,0)

        h2[0,1] = 1        
        res = OptimizationAttacker.hash_norm(h1,h2)
        self.assertEqual(res,1)

        h2[1,1] = 1        
        res = OptimizationAttacker.hash_norm(h1,h2)
        self.assertEqual(res, np.sqrt(2))


if __name__ == '__main__':
    unittest.main()
