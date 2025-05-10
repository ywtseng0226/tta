"""
Adapted from: https://github.com/BIT-DA/RoTTA/blob/main/core/data/ttasampler.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.patches as mpatches

from torch.utils.data.sampler import Sampler
from src.data.base_dataset import DatumBase
from typing import List
from collections import defaultdict
from numpy.random import dirichlet
from math import exp

class LabelDirichletDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase], gamma, batch_size, slots=None):

        self.domain_dict = defaultdict(list)
        self.classes = set()
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
            self.classes.add(item.label)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_class = len(self.classes)
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []
        for domain in self.domains:
            indices = np.array(self.domain_dict[domain])
            labels = np.array([self.data_source[i].label for i in indices])

            class_indices = [np.argwhere(labels == y).flatten() for y in range(self.num_class)]
            slot_indices = [[] for _ in range(self.num_slots)]

            label_distribution = dirichlet([self.gamma] * self.num_slots, self.num_class)

            for c_ids, partition in zip(class_indices, label_distribution):
                for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int))):
                    slot_indices[s].append(ids)

            for s_ids in slot_indices:
                permutation = np.random.permutation(range(len(s_ids)))
                ids = []
                for i in permutation:
                    ids.extend(s_ids[i])
                final_indices.extend(indices[ids])

        return iter(final_indices)

def plot_batch_distribution(corruption_types, corruption_counts_list, corruption_color_map, output_dir=".", save_name="batch-distribution.png"):
    """
    Plot stacked bar showing the distribution of 15 corruption types across batches,
    and add legend to indicate color -> corruption mapping.

    Args:
        corruption_counts_list (list of np.array): list where each element is shape (15,)
        corruption_color_map (dict): corruption_id -> RGB color
        output_dir (str): folder to save figure
        save_name (str): file name to save
    """
    corruption_counts_array = np.array(corruption_counts_list)  # shape [num_batches, 15]
    num_batches = corruption_counts_array.shape[0]
    num_corruptions = corruption_counts_array.shape[1]
    
    x = np.arange(num_batches)  # batch index
    bottoms = np.zeros(num_batches)  # starting bottom for each batch
    
    plt.figure(figsize=(18, 6))
    
    for corruption_id in range(num_corruptions):
        heights = corruption_counts_array[:, corruption_id]
        plt.bar(
            x, heights, bottom=bottoms,
            color=corruption_color_map[corruption_id],
            width=1.0,
            edgecolor='none'
        )
        bottoms += heights  # Update bottom for next corruption
        
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.title("Corruption Type Distribution Across Batches", fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # --- NEW: Add Legend for Corruption Types ---
    legend_patches = []
    for corruption_id, color in corruption_color_map.items():
        patch = mpatches.Patch(color=color, label=corruption_types[corruption_id])
        legend_patches.append(patch)
    
    plt.legend(handles=legend_patches, title="Corruption Types", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")  # Use tight bbox to include legend
    plt.close()
    print(f"[✓] Saved batch distribution plot to: {save_path}")

class GaussianCGSSampler(Sampler):
    def __init__(self, data_source, corruption_types, batch_size=64, sigma=5000):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sigma = sigma
        self.domain_dict = defaultdict(list)
        self.domain_centers = {}

        # Group sample indices by domain
        for idx, item in enumerate(data_source):
            self.domain_dict[item.domain].append(idx)

        self.domains = sorted(self.domain_dict.keys())
        self.num_domains = len(self.domains)
        self.total_samples = sum(len(v) for v in self.domain_dict.values())

        # Assume each domain has same number of samples, and center positions
        self.samples_per_domain = len(self.domain_dict[self.domains[0]])
        self.domain_centers = {d: i * self.samples_per_domain for i, d in enumerate(self.domains)}

        # Build index pools (no replacement sampling initially)
        self.corruption_types = corruption_types
        self.index_pool = {d: list(self.domain_dict[d]) for d in self.domains}
        for v in self.index_pool.values():
            np.random.shuffle(v)

        self.corruption_counts_list = []

    def gaussian(self, x, mu, sigma):
        return exp(- ((x - mu) ** 2) / (2 * sigma ** 2))

    def __iter__(self):
        pointer = 0
        final_indices = []

        while pointer + self.batch_size <= self.total_samples:
            # Compute Gaussian weights based on distance from each domain center
            weights = []
            for d in self.domains:
                w = self.gaussian(pointer, self.domain_centers[d], self.sigma)
                weights.append(w)
            weights = np.array(weights)
            weights /= weights.sum()

            # Assign sample count to each domain
            domain_counts = (weights * self.batch_size).astype(int)

            # Correction to ensure total = batch size
            diff = self.batch_size - domain_counts.sum()
            if diff > 0:
                domain_counts[np.argmax(weights)] += diff
            elif diff < 0:
                domain_counts[np.argmax(weights)] -= abs(diff)

            # Draw samples
            batch_indices = []
            for d, count in zip(self.domains, domain_counts):
                pool = self.index_pool[d]
                if len(pool) < count:
                    continue  # skip if not enough data
                batch_indices += pool[:count]
                # self.index_pool[d] = pool[count:]  # (you commented out here, so samples are **not removed**)

            if len(batch_indices) == self.batch_size:
                final_indices.extend(batch_indices)
                pointer += self.batch_size
            else:
                break
        
        # corruption_color_map = {i: c for i, c in enumerate(plt.get_cmap('tab20').colors[:15])}
        # plot_batch_distribution(self.corruption_types, self.corruption_counts_list[:5000], corruption_color_map, "./materials/")
        # exit()
        return iter(final_indices)

    def __len__(self):
        return len(self.data_source)


class GaussianCGSSamplerV2(Sampler):
    """
    A Gaussian-based sampler for Continual Gradual Shifting (CGS) settings in Test-Time Adaptation.
    Ensures that each batch is fully filled even if some domains run out of samples, 
    by reshuffling domain pools as needed.
    """

    def __init__(self, data_source, batch_size=64, sigma=8000):
        """
        Args:
            data_source (list): A list of DatumBase objects with 'domain' attributes.
            batch_size (int): The batch size.
            sigma (float): Standard deviation for the Gaussian kernel.
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.sigma = sigma

        self.domain_dict = defaultdict(list)
        self.index_pool = {}
        self.domain_centers = {}

        # Group sample indices by domain
        for idx, item in enumerate(data_source):
            self.domain_dict[item.domain].append(idx)

        self.domains = sorted(self.domain_dict.keys())
        self.num_domains = len(self.domains)
        self.samples_per_domain = len(self.domain_dict[self.domains[0]])
        self.total_samples = sum(len(v) for v in self.domain_dict.values())

        # Set the "center" position for each domain on the time axis
        self.domain_centers = {
            d: i * self.samples_per_domain + self.samples_per_domain // 2
            for i, d in enumerate(self.domains)
        }

        # Shuffle samples for each domain initially
        self.index_pool = {
            d: np.random.permutation(indices).tolist()
            for d, indices in self.domain_dict.items()
        }

    def gaussian(self, x, mu, sigma):
        """Compute Gaussian weight centered at mu for position x."""
        return exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def __iter__(self):
        pointer = 0
        final_indices = []

        # Estimate maximum number of iterations
        max_iters = self.total_samples // self.batch_size

        for _ in range(max_iters):
            # Compute Gaussian weights for the current pointer
            weights = np.array([
                self.gaussian(pointer, self.domain_centers[d], self.sigma)
                for d in self.domains
            ])
            weights /= weights.sum()  # normalize to sum=1

            # Allocate samples according to Gaussian weights
            domain_counts = (weights * self.batch_size).astype(int)

            # Adjust if total counts mismatch batch_size
            diff = self.batch_size - domain_counts.sum()
            if diff != 0:
                domain_counts[np.argmax(weights)] += diff

            batch_indices = []

            # Sample according to domain counts
            for d, count in zip(self.domains, domain_counts):
                pool = self.index_pool[d]

                # If insufficient samples in domain, reshuffle
                if len(pool) < count:
                    pool = np.random.permutation(self.domain_dict[d]).tolist()
                    self.index_pool[d] = pool

                batch_indices.extend(pool[:count])
                self.index_pool[d] = pool[count:]  # consume used samples

            assert len(batch_indices) == self.batch_size, "Batch is not fully filled!"
            final_indices.extend(batch_indices)
            pointer += self.batch_size

        return iter(final_indices)

    def __len__(self):
        return len(self.data_source)
  
class DirichletEpisodicDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase], gamma, batch_size, slots=5):

        self.domain_dict = defaultdict(list)
        self.label_dict = defaultdict(list)
        self.classes = set()
        
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)

        # Randomly shuffling items in each domain group
        for v in self.domain_dict:
            np.random.shuffle(self.domain_dict[v])

        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_class = len(self.classes)
        self.num_episode = 5
        if slots is not None:
            self.num_slots = slots
        else:
            self.num_slots = self.num_class if self.num_class <= 100 else 100

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []    
        domain_idcs = [self.domain_dict[y] for y in self.domains]
        slot_indices = [[] for _ in range(self.num_slots)]
        domain_distribution = np.random.dirichlet([self.gamma] * self.num_slots, len(self.domains))
        
        for c_ids, partition in zip(domain_idcs, domain_distribution):
            for s, ids in enumerate(np.split(c_ids, (np.cumsum(partition)[:-1] * len(c_ids)).astype(int))):
                slot_indices[s].append(ids)
        final_indices = []
        for s_ids in slot_indices:
            permutation = np.random.permutation(range(len(s_ids)))
            ids = []
            for i in permutation:
                ids.extend(s_ids[i])
            final_indices.extend(ids)
        
        return iter(final_indices)

class TemporalDomainSequence(Sampler):
    def __init__(self, data_source: List[DatumBase]):

        self.domain_dict = defaultdict(list)
        self.label_dict = defaultdict(list)
        self.classes = set()
        
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)

        # Randomly shuffling items in each domain group
        for v in self.domain_dict:
            np.random.shuffle(self.domain_dict[v])

        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        self.data_source = data_source
        self.num_class = len(self.classes)
        
    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        final_indices = []
        for y in self.domains:
            final_indices.extend(self.domain_dict[y])        
        return iter(final_indices)
    
def build_sampler(cfg, data_source: List[DatumBase], **kwargs):
    if cfg.LOADER.SAMPLER.TYPE == "class_temporal":
        return LabelDirichletDomainSequence(data_source, cfg.LOADER.SAMPLER.GAMMA, cfg.TEST.BATCH_SIZE, **kwargs)
    elif cfg.LOADER.SAMPLER.TYPE == "cgs":
        return GaussianCGSSampler(data_source, cfg.CORRUPTION.TYPE)
    else:
        raise NotImplementedError()
