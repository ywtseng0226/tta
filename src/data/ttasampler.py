"""
Adapted from: https://github.com/BIT-DA/RoTTA/blob/main/core/data/ttasampler.py
"""

import numpy as np
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

class GaussianCGSSampler(Sampler):
    def __init__(self, data_source, batch_size=64, sigma=8000):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sigma = sigma
        self.domain_dict = defaultdict(list)
        self.domain_centers = {}

        # 假設每個 domain 有相同數量的 samples，且順序為 domain0, domain1, ...
        for idx, item in enumerate(data_source):
            self.domain_dict[item.domain].append(idx)

        self.domains = sorted(self.domain_dict.keys())
        self.num_domains = len(self.domains)
        self.total_samples = sum(len(v) for v in self.domain_dict.values())

        # domain_i 的中心點位置
        self.samples_per_domain = len(self.domain_dict[self.domains[0]])
        self.domain_centers = {d: i * self.samples_per_domain for i, d in enumerate(self.domains)}

        # 為每個 domain 建立 index queue（不重複使用）
        self.index_pool = {d: list(self.domain_dict[d]) for d in self.domains}
        for v in self.index_pool.values():
            np.random.shuffle(v)

    def gaussian(self, x, mu, sigma):
        return exp(- ((x - mu) ** 2) / (2 * sigma ** 2))

    def __iter__(self):
        pointer = 0
        final_indices = []

        while pointer + self.batch_size <= self.total_samples:
            # 計算每個 domain 對當前 pointer 的機率
            weights = []
            for d in self.domains:
                w = self.gaussian(pointer, self.domain_centers[d], self.sigma)
                weights.append(w)
            weights = np.array(weights)
            weights /= weights.sum()  # normalize

            # 根據機率分配每個 domain 的 sample 數
            domain_counts = (weights * self.batch_size).astype(int)

            # 修正總和以符合 batch size
            diff = self.batch_size - domain_counts.sum()
            if diff > 0:
                domain_counts[np.argmax(weights)] += diff
            elif diff < 0:
                domain_counts[np.argmax(weights)] -= abs(diff)

            # 從每個 domain 抽 sample
            batch_indices = []
            for d, count in zip(self.domains, domain_counts):
                pool = self.index_pool[d]
                if len(pool) < count:
                    continue  # skip if not enough data left
                batch_indices += pool[:count]
                # self.index_pool[d] = pool[count:]

            if len(batch_indices) == self.batch_size:
                final_indices.extend(batch_indices)
                pointer += self.batch_size
            else:
                break  # not enough data left

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
        return GaussianCGSSampler(data_source)
    else:
        raise NotImplementedError()
