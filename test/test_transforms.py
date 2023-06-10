import unittest
from buildings_bench import transforms
from pathlib import Path
import os
import numpy as np
import torch
import pandas as pd


class TestStandardScaler(unittest.TestCase):
    def setUp(self):

        self.ss = transforms.StandardScalerTransform()
        save_dir = os.environ.get('BUILDINGS_BENCH', '')
        self.ss.load(Path(save_dir) / 'metadata' / 'transforms')


    def test_load_standard_scaler(self):

    
        self.assertIsNotNone(self.ss.mean_, True)
        self.assertIsNotNone(self.ss.std_, True)


    def test_standard_scale(self):
        
        x = torch.FloatTensor([[100.234], [0.234], [55.523]])
    
        y = self.ss.transform(x)
        z = self.ss.undo_transform(y)

        self.assertTrue(torch.allclose(x, z, atol=1e-3))


class TestBoxCox(unittest.TestCase):
    def setUp(self):

        self.bc = transforms.BoxCoxTransform()
        metadata_dir = os.environ.get('BUILDINGS_BENCH', '')
        self.bc.load(Path(metadata_dir) / 'metadata' / 'transforms')


    def test_load_boxcox(self):
        self.assertIsNotNone(self.bc.boxcox.lambdas_, True)


    def test_boxcox(self):
            
        x = torch.FloatTensor([[100.234], [0.234], [55.523]])
    
        y = self.bc.transform(x)
        z = self.bc.undo_transform(torch.from_numpy(y).float())

        # assert allclose
        self.assertTrue(torch.allclose(x, z, atol=1e-3))


class TestLatLonTransform(unittest.TestCase):
    def setUp(self):
        self.ll = transforms.LatLonTransform()

    def test_load_latlon(self):
        self.assertIsNotNone(self.ll.lat_means, True)
        self.assertIsNotNone(self.ll.lon_means, True)
        self.assertIsNotNone(self.ll.lat_stds, True)
        self.assertIsNotNone(self.ll.lon_stds, True)

    def test_latlon(self):
        x = np.array([[100.234, 0.234], [0.234, 55.523], [55.523, 100.234]])
        y = self.ll.transform_latlon(x)
        z = self.ll.undo_transform(y)

        self.assertTrue(np.allclose(x, z, atol=1e-3))


class TestTimestampTransform(unittest.TestCase):
    def setUp(self):
        self.tt = transforms.TimestampTransform()

    def test_timestamp(self):
        x = np.array(['2016-01-01 00:00:00', '2016-01-01 01:00:00'])
        # convert x to dataframe
        x = pd.DataFrame(x, columns=['timestamp'])
        y = self.tt.transform(x.timestamp)
        z = self.tt.undo_transform(y)

        #print(x,y,z)
        self.assertEqual(z[0,0], pd.to_datetime(x.timestamp).dt.dayofyear.values[0])
        self.assertEqual(z[0,1], pd.to_datetime(x.timestamp).dt.dayofweek.values[0])
        self.assertEqual(z[0,2], pd.to_datetime(x.timestamp).dt.hour.values[0])


