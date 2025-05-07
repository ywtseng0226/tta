import torch
import torch.nn.functional as F
import math
import random

from copy import deepcopy
from torch import nn
from collections import Counter

# MemoryItem: stores one sample and its meta info in memory bank
class MemoryItem:
    def __init__(self, data=None, uncert=0, age=0, label=None):
        self.data = data
        self.uncert = uncert
        self.age = age
        self.label = label

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncert, self.age

    def empty(self):
        return self.data == "empty"

# MyTTAMemory: memory bank for adaptive sample selection at test time
class ShortTermMemory:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0, lambda_d=1.0,
                 max_bank_num=1, eta=0.1, base_threshold=0.4, ):
        self.capacity = capacity  # total memory capacity
        self.num_class = num_class
        self.per_class = capacity / num_class  # class-wise quota
        self.lambda_t = lambda_t  # age factor
        self.lambda_u = lambda_u  # uncertainty factor
        self.lambda_d = lambda_d  # distance factor
        self.max_bank_num = max_bank_num  # max number of banks
        self.eta = eta
        self.base_threshold = base_threshold
        self.banks = []  # memory banks (clusters)
        self.num_consolidations = 0

    # Compute feature descriptor for a single image: (mean, var) over channels
    def compute_instance_descriptor(self, data):
        data_mean = torch.mean(data, dim=(1, 2))
        data_var = torch.var(data, dim=(1, 2))
        return data_mean, data_var

    def update_bank_descriptor(self, bank):
        all_items = []
        for class_items in bank["items"]:
            all_items.extend(class_items)
        if len(all_items) == 0:
            bank["descriptor"] = (None, None)
            return
        data_tensor = torch.stack([item.data for item in all_items])
        bank_mean = torch.mean(data_tensor, dim=(0, 2, 3))
        bank_var = torch.var(data_tensor, dim=(0, 2, 3))
        bank["descriptor"] = (bank_mean, bank_var)
    
    # Euclidean distance between two (mean, var) descriptors
    def descriptor_distance(self, instance_descriptor, bank_descriptor):
        inst_mean, inst_var = instance_descriptor
        bank_mean, bank_var = bank_descriptor
        inst_concat = torch.cat([inst_mean, inst_var])
        bank_concat = torch.cat([bank_mean, bank_var])
        return torch.norm(inst_concat - bank_concat, p=2)

    # Get adaptive threshold for deciding if a sample fits into a bank
    def get_dynamic_threshold(self, bank):
        bank_descriptor = bank["descriptor"]
        if bank_descriptor[0] is None or bank_descriptor[1] is None:
            return self.base_threshold
        std = torch.sqrt(bank_descriptor[1])
        avg_std = torch.mean(std).item()
        return self.base_threshold * (1 + avg_std)

    def consolidation(self):
        self.num_consolidations += 1
        min_dist = float("inf")
        pair_to_merge = None

        for i in range(len(self.banks)):
            for j in range(i + 1, len(self.banks)):
                desc_i = self.banks[i]["descriptor"]
                desc_j = self.banks[j]["descriptor"]
                if desc_i[0] is None or desc_j[0] is None:
                    continue
                dist = self.descriptor_distance(desc_i, desc_j)
                if dist < min_dist:
                    min_dist = dist
                    pair_to_merge = (i, j)

        if pair_to_merge is None:
            return

        i, j = pair_to_merge
        bank_i = self.banks[i]
        bank_j = self.banks[j]

        for cls in range(self.num_class):
            bank_i["items"][cls].extend(bank_j["items"][cls])

        del self.banks[j]

        all_items = []
        for cls_items in bank_i["items"]:
            all_items.extend(cls_items)

        if len(all_items) > self.capacity:
            all_items.sort(key=lambda item: item.uncert)
            all_items = all_items[:self.capacity]

        new_class_items = [[] for _ in range(self.num_class)]
        for item in all_items:
            label = item.label
            if len(new_class_items[label]) < self.per_class:
                new_class_items[label].append(item)

        bank_i["items"] = new_class_items
        self.update_bank_descriptor(bank_i)

    def remove_instance(self, bank, cls, new_score):
        """
        Decide whether there's space to insert a new item in a class.
        If not, determine whether to replace an existing item.
        """
        items = bank["items"][cls]
        class_occupied = len(items)
        all_occupancy = sum(len(lst) for lst in bank["items"])

        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                # Too full, consider replacing from majority classes
                max_count = max(len(lst) for lst in bank["items"])
                majority_classes = [i for i, lst in enumerate(bank["items"]) if len(lst) == max_count]
                return self.remove_from_classes(bank, majority_classes, new_score)
        else:
            # Class full, try to replace within the same class
            return self.remove_from_classes(bank, [cls], new_score)

    def remove_from_classes(self, bank, classes, score_base):
        max_score = None
        max_class = None
        max_index = None

        for c in classes:
            items = bank["items"][c]
            for idx, item in enumerate(items):
                s = self.heuristic_score(item.age, item.uncert, item.data, bank)
                if max_score is None or s > max_score:
                    max_score = s
                    max_class = c
                    max_index = idx

        if max_class is not None and max_score > score_base:
            bank["items"][max_class].pop(max_index)
            return True
        else:
            return False

    def add_age(self, bank):
        for class_items in bank["items"]:
            for item in class_items:
                item.increase_age()

    def heuristic_score(self, age, uncert, data, bank):
        # Lower score means higher priority to be kept
        instance_descriptor = self.compute_instance_descriptor(data)
        bank_descriptor = bank["descriptor"]
        distance = self.descriptor_distance(instance_descriptor, bank_descriptor)
        score = self.lambda_t * (1 / (1 + math.exp(-age / self.capacity))) + \
                self.lambda_u * (uncert / math.log(self.num_class)) + \
                self.lambda_d * distance
        return score

    def add_instance(self, instance):
        x, pred, uncert, label = instance
        new_item = MemoryItem(data=x, uncert=uncert, age=0, label=label)
        instance_descriptor = self.compute_instance_descriptor(x)

        # Find the closest matching bank
        target_bank, target_distance = None, float('inf')
        for bank in self.banks:
            bank_descriptor = bank["descriptor"]
            dist = self.descriptor_distance(instance_descriptor, bank_descriptor)
            if dist < target_distance:
                target_distance = dist
                target_bank = bank
        
        # Do we need to add a new bank?
        if target_bank is None or (target_distance > self.get_dynamic_threshold(target_bank)):
            new_bank = {
                    "items": [[] for _ in range(self.num_class)],
                    "descriptor": instance_descriptor
                }
            self.banks.append(new_bank)
            target_bank = new_bank
            if len(self.banks) > self.max_bank_num: 
                self.consolidation()
        
        # Score the item and attempt to insert
        class_idx = pred
        new_score = self.heuristic_score(age=0, uncert=uncert, data=x, bank=target_bank)

        if self.remove_instance(target_bank, class_idx, new_score):
            target_bank["items"][class_idx].append(new_item)
            self.update_bank_descriptor(target_bank)

        # Increment age of all samples in the target bank
        self.add_age(target_bank)

    def get_sup_data(self, batch_samples, topk=3, max_samples=64):
        """
        Select top-K banks based on total distance to all samples in the batch,
        then sample up to `max_samples` for learning.

        Args:
            batch_samples: Tensor of shape (N, C, H, W)
            topk: number of closest banks to use
            max_samples: upper limit of replay samples (default = 64)

        Returns:
            sup_data: list of tensors
            sup_age: list of corresponding ages
        """
        if not self.banks:
            return [], []

        # Compute descriptor (mean, var) for each sample
        sample_descriptors = []
        for i in range(batch_samples.shape[0]):
            mean = torch.mean(batch_samples[i], dim=(1, 2))  # (C,)
            var = torch.var(batch_samples[i], dim=(1, 2))    # (C,)
            sample_descriptors.append((mean, var))

        # Compute total distance to each bank
        total_dists = []
        valid_bank_ids = []

        for bank_id, bank in enumerate(self.banks):
            bank_desc = bank["descriptor"]
            if bank_desc[0] is None:
                continue

            dist_sum = sum(self.descriptor_distance(sample_desc, bank_desc) for sample_desc in sample_descriptors)
            total_dists.append(dist_sum)
            valid_bank_ids.append(bank_id)

        if not total_dists:
            return [], []

        # Get top-K banks by lowest total distance
        sorted_ids = sorted(zip(valid_bank_ids, total_dists), key=lambda x: x[1])
        topk_bank_ids = [bank_id for bank_id, _ in sorted_ids[:topk]]

        # ollect memory items from selected banks
        selected_items = []
        for bank_id in topk_bank_ids:
            for cls_items in self.banks[bank_id]["items"]:
                selected_items.extend(cls_items)

        # Limit to max_samples with random shuffle
        random.shuffle(selected_items)
        selected_items = selected_items[:max_samples]

        # Build outputs
        sup_data = [item.data for item in selected_items]
        sup_age = [item.age for item in selected_items]

        return sup_data, sup_age

