import unittest
import pandas as pd
import random
import math

class TestRollingMeans(unittest.TestCase):

    def setUp(self):
        self.grid = pd.read_pickle('data/grid_features.pkl')
        self.grid = self.grid[(self.grid['d'] < 1942) & (self.grid['d'] > 1942-180)]
        self.rms = [7, 14, 30, 60]
        self.tolerance = 1e-3

    def test_random_rolling_means(self):
        rnd_id = random.choice(self.grid['id'].unique())
        rnd_day = random.choice(self.grid['d'].unique())
        
        for rm in self.rms:
            n = self.grid[(self.grid['d'] >= rnd_day-(rm-1)) & (self.grid['d'] <= rnd_day) & (self.grid['id'] == rnd_id)]['sales'].mean()
            t_n = self.grid[(self.grid['d'] == rnd_day) & (self.grid['id'] == rnd_id)][f'rm_{rm}'].values[0]
            
            self.assertTrue(math.isclose(n, t_n, rel_tol=self.tolerance),
                            f"True value {n:.4f} and Predicted value {t_n:.4f} are not close enough for rm: {rm}.")

if __name__ == "__main__":
    unittest.main()
