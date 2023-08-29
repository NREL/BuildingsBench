import faiss
import faiss.contrib.torch_utils 
from typing import Union
import numpy as np
from pathlib import Path
import torch


class LoadQuantizer:
    """Quantize load timeseries with KMeans. Merge centroids that are within a threshold."""
    def __init__(self, seed: int = 1, num_centroids: int = 2274,
                       with_merge=False, merge_threshold=0.01, device: str = 'cpu'):
        """
        Args:
            seed (int): random seed. Default: 1.
            num_centroids (int): number of centroids: Default: 2274.
            with_merge (bool): whether to merge centroids that are within a threshold: Default: False.
            merge_threshold (float): threshold for merging centroids. Default: 0.01 (kWh).
            device (str): cpu or cuda. Default: cpu.
        """
        self.seed = seed
        self.K = num_centroids
        self.with_merge = with_merge
        self.merge_threshold = merge_threshold
        self.kmeans = None
        self.original_order_to_sorted_centroid_indices = None
        self.original_order_to_merged_centroid_map = None
        self.merged_centroids = None
        self.device = device

    def get_vocab_size(self) -> int:
        if self.with_merge:
            return len(self.merged_centroids)
        return self.K

    def train(self, sample: np.ndarray) -> None:
        """Fit KMeans to a subset of the data. 
        
        Optionally, merge centroids that are within a threshold.

        Args:
            sample (np.ndarray): shape [num_samples, 1]
        """
        self.kmeans = faiss.Kmeans(d=1, k=self.K, niter=300, nredo=10,
                     seed=self.seed, verbose=True, gpu=(True if 'cuda' in self.device else False))
        # max is 256, min is 39 per centroid by defaul
        # https://github.com/facebookresearch/faiss/blob/d5ca6f0a79aa382bb98d2221a8f61c1a200efa25/faiss/Clustering.cpp#L2t
        self.kmeans.train(sample.reshape(-1,1))
        
        if self.with_merge:
            # sort the centroids and return the sorted indices
            sorted_indices = np.argsort(self.kmeans.centroids.reshape(-1))
            # sort the centroids
            #unsorted_index_to_value_map = self.kmeans.centroids.copy()
            index_to_value_map = self.kmeans.centroids[sorted_indices]
            # create a map from the sorted indices to the original indices
            self.original_order_to_sorted_centroid_indices = np.zeros_like(sorted_indices)
            for i in range(len(sorted_indices)):
                self.original_order_to_sorted_centroid_indices[sorted_indices[i]] = i

            # Iterate over sorted centroids, merge centroids that are within a threshold
            previous_centroid_value = index_to_value_map[0]
            merge_classes = [0]
            candidate_centroids = [previous_centroid_value]

            # new list of centroids
            self.merged_centroids = []
            # list that maps the original centroid indices to the new centroid indices
            candidate_index_list = []


            #  Iterate over the sorted centroids.
            for i in range(1,len(index_to_value_map)):
                # If the current centroid is within a threshold of the previous centroid,
                # add it to the merged class. 
                if index_to_value_map[i] - previous_centroid_value < self.merge_threshold:
                    merge_classes += [i]
                    candidate_centroids += [index_to_value_map[i]]
                #  If the current centroid is not within a threshold of the previous centroid,
                #  merge the set of candidate centroids, add the merged class indices to the
                # candidate list, and update previous centroid value to point to current index
                else:
                    # I take the average of the centroids
                    # being merged
                    self.merged_centroids += [np.mean(candidate_centroids)]
                    candidate_index_list += [merge_classes]
                    # reset this with current index
                    merge_classes = [i]
                    candidate_centroids = [index_to_value_map[i]]
                    previous_centroid_value = index_to_value_map[i]
            # Add the last centroid to the new list of centroids and add the merged class to the merge list.
            self.merged_centroids += [np.mean(candidate_centroids)]
            candidate_index_list += [merge_classes]
            self.merged_centroids = np.array(self.merged_centroids)
            # Create a map from the original centroid index to the new centroid index
            self.original_order_to_merged_centroid_map = np.zeros(len(index_to_value_map), dtype=np.int32)
            for i in range(len(candidate_index_list)):
                for j in candidate_index_list[i]:
                    self.original_order_to_merged_centroid_map[j] = i
            print('Merged {} centroids to {} centroids'.format(len(index_to_value_map), len(self.merged_centroids)))
            self.K = len(self.merged_centroids)
            
    def save(self, output_path: Path) -> None:
        if 'cuda' in self.device:
            self.kmeans.index = faiss.index_gpu_to_cpu(self.kmeans.index)

        chunk = faiss.serialize_index(self.kmeans.index)
        np.save(output_path / "kmeans_K={}.npy".format(self.K), chunk)
        np.save(output_path / "kmeans_centroids_K={}.npy".format(self.K), self.kmeans.centroids)
        if self.with_merge:
            np.save(output_path / "kmeans_original_to_sorted_indices_K={}.npy".format(self.K),
                    self.original_order_to_sorted_centroid_indices)
            np.save(output_path / "kmeans_original_to_merged_map_K={}.npy".format(self.K),
                    self.original_order_to_merged_centroid_map)
            np.save(output_path / "kmeans_merged_centers_K={}.npy".format(self.K),
                    self.merged_centroids)

    def load(self, saved_path: Path) -> None:
        chunk = np.load(
            saved_path / "kmeans_K={}.npy".format(self.K), allow_pickle=True)
        self.kmeans = faiss.Kmeans(d=1, k=self.K, niter=200, nredo=20,
                     seed=self.seed, verbose=True, gpu=(True if 'cuda' in self.device else False))
        self.kmeans.index = faiss.deserialize_index(chunk)
        self.kmeans.centroids = np.load(
            saved_path / "kmeans_centroids_K={}.npy".format(self.K), allow_pickle=True)

        if self.with_merge:
            self.original_order_to_sorted_centroid_indices = np.load(
                saved_path / "kmeans_original_to_sorted_indices_K={}.npy".format(self.K), allow_pickle=True)
            self.original_order_to_merged_centroid_map = np.load(
                saved_path / "kmeans_original_to_merged_map_K={}.npy".format(self.K), allow_pickle=True)
            self.merged_centroids = np.load(
                saved_path / "kmeans_merged_centers_K={}.npy".format(self.K), allow_pickle=True)
            
            print(f'Loaded Kmeans quantizer with K={len(self.merged_centroids)}')
        else:
            print(f'Loaded Kmeans quantizer with K={self.K}')

        if 'cuda' in self.device:
            local_rank = int(self.device.split(':')[-1])
            faiss_res = faiss.StandardGpuResources()
            self.kmeans.index = faiss.index_cpu_to_gpu(faiss_res, local_rank, self.kmeans.index)
            self.kmeans.centroids = torch.from_numpy(self.kmeans.centroids).float().to(self.device)
            self.kmeans.centroids.squeeze()
            if self.with_merge:
                self.original_order_to_sorted_centroid_indices = torch.from_numpy( 
                    self.original_order_to_sorted_centroid_indices).long().to(self.device)
                self.original_order_to_merged_centroid_map = torch.from_numpy(
                    self.original_order_to_merged_centroid_map).long().to(self.device)
                self.merged_centroids = torch.from_numpy(
                    self.merged_centroids).float().to(self.device)
                self.merged_centroids.squeeze()
            print(f'Kmeans quantizer moved to GPU: {type(self.kmeans.index)}')


    def transform(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Quantize a sample of load values into a sequence of indices.
        
        Args:
            sample Union[np.ndarray, torch.Tensor]: of shape (n, 1) or (b,n,1). 
                type is numpy if device is cpu or torch Tensor if device is cuda.

        Returns:
            sample (Union[np.ndarray, torch.Tensor]): of shape (n, 1) or (b,n,1).
        """
        init_shape = sample.shape
        sample = sample.reshape(-1,1)
        sample = self.kmeans.index.search(sample, 1)[1].reshape(-1)
        if self.with_merge:
            sample = self.original_order_to_sorted_centroid_indices[sample]
            sample = self.original_order_to_merged_centroid_map[sample]
        sample = sample.reshape(init_shape)
        return sample


    def undo_transform(self, sample: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Dequantize a sample of integer indices into a sequence of load values.

        Args:
            sample Union[np.ndarray, torch.Tensor]: of shape (n, 1) or (b,n,1). 
                type is numpy if device is cpu or torch Tensor if device is cuda.

        Returns:
            sample (Union[np.ndarray, torch.Tensor]): of shape (n, 1) or (b,n,1).
        """
        init_shape = sample.shape
        sample = sample.reshape(-1)
        if self.with_merge:
            sample = self.merged_centroids[sample]
        else:
            sample = self.kmeans.centroids[sample]
        sample = sample.reshape(init_shape)
        return sample
