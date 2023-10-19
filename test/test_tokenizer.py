import unittest
from buildings_bench import tokenizer
import os
from pathlib import Path 
import numpy as np


class TestTokenizer(unittest.TestCase):
    """ Test the KMeans tokenizer"""

    def test_load(self):
        """ Test loading the tokenizer """

        transform_path = Path(os.environ.get('BUILDINGS_BENCH', '')) \
                              / 'metadata' / 'transforms'

        load_quantizer = tokenizer.LoadQuantizer(with_merge=True, num_centroids=3747, device='cpu')
        load_quantizer.load(transform_path)
        self.assertEqual(load_quantizer.get_vocab_size(), 3747)


    def test_transform_cpu(self):
        """ Test the transform method on CPU."""

        transform_path = Path(os.environ.get('BUILDINGS_BENCH', '')) \
                              / 'metadata' / 'transforms'

        load_quantizer = tokenizer.LoadQuantizer(with_merge=True, num_centroids=3747, device='cpu')
        load_quantizer.load(transform_path)

        x = np.array([[100.234], [0.234], [55.523]])
        y = load_quantizer.transform(x)
        z = load_quantizer.undo_transform(y)

        #print(x,y,z)

        self.assertTrue(np.allclose(x, z, atol=1))
