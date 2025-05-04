import torch
import torch.nn.functional as F
import math
import random

from copy import deepcopy
from torch import nn
from collections import Counter


# Compute class-wise mean features given pseudo labels
def compute_feat_mean(feats, pseudo_lbls):
    lbl_uniq = torch.unique(pseudo_lbls)
    lbl_group = [torch.where(pseudo_lbls == l)[0] for l in lbl_uniq]
    group_avgs = []
    for i, lbl_idcs in enumerate(lbl_group):
        group_avgs.append(feats[lbl_idcs].mean(axis=0).unsqueeze(0))
    return lbl_uniq, group_avgs 

# DivergenceScore: measure how much current features deviate from source prototypes
class DivergenceScore(nn.Module):
    def __init__(self, src_prototype, src_prototype_cov):
        super().__init__()
        self.src_proto = src_prototype  # shape: [num_classes, feat_dim]
        self.src_proto_cov = src_prototype_cov  # shape: [num_classes, feat_dim]

        # Gaussian scaled squared loss
        def GSSLoss(input, target, target_cov):
            return ((input - target).pow(2) / (target_cov + 1e-6)).mean()

        self.lss = GSSLoss

    def forward(self, feats, pseudo_lbls):
        # Compute mean feature for each predicted class
        lbl_uniq, group_avgs = compute_feat_mean(feats, pseudo_lbls)
        return self.lss(
            torch.cat(group_avgs, dim=0),
            self.src_proto[lbl_uniq],
            self.src_proto_cov[lbl_uniq],
        )

# PrototypeMemory: maintains moving average feature prototypes per class
class PrototypeMemory:
    def __init__(self, src_prototype, num_classes) -> None:
        self.src_proto = src_prototype.squeeze(1)  # initial source prototype per class
        self.mem_proto = deepcopy(self.src_proto)  # moving average memory
        self.num_classes = num_classes
        self.src_proto_l2 = torch.cdist(self.src_proto, self.src_proto, p=2)

    # Update memory using new target-domain features
    def update(self, feats, pseudo_lbls, nu=0.05):
        lbl_uniq = torch.unique(pseudo_lbls)
        lbl_group = [torch.where(pseudo_lbls == l)[0] for l in lbl_uniq]
        for i, lbl_idcs in enumerate(lbl_group):
            psd_lbl = lbl_uniq[i]
            batch_avg = feats[lbl_idcs].mean(axis=0)
            # Exponential moving average update
            self.mem_proto[psd_lbl] = (1 - nu) * self.mem_proto[psd_lbl] + nu * batch_avg

    def get_mem_prototype(self):
        return self.mem_proto

# MemoryItem: stores one sample and its meta info in memory bank
class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0, true_label=None):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age
        self.true_label = true_label

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"

# MyTTAMemory: memory bank for adaptive sample selection at test time
class MyTTAMemory:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0, lambda_d=1.0,
                 mem_num=5, eta=0.1, base_threshold=0.5, ):
        self.capacity = capacity  # total memory capacity
        self.num_class = num_class
        self.per_class = capacity / num_class  # class-wise quota
        self.lambda_t = lambda_t  # age factor
        self.lambda_u = lambda_u  # uncertainty factor
        self.lambda_d = lambda_d  # distance factor
        self.mem_num = mem_num  # max number of banks
        self.eta = eta
        self.base_threshold = base_threshold
        self.banks = []  # memory banks (clusters)

    # Compute feature descriptor for a single image: (mean, var) over channels
    def compute_instance_descriptor(self, data):
        data_mean = torch.mean(data, dim=(1, 2))
        data_var = torch.var(data, dim=(1, 2))
        return data_mean, data_var

    # Compute bank-wide descriptor by aggregating all items
    def compute_bank_descriptor(self, bank_items):
        all_items = []
        for class_items in bank_items:
            all_items.extend(class_items)
        if len(all_items) == 0:
            return None, None
        data_tensor = torch.stack([item.data for item in all_items])
        bank_mean = torch.mean(data_tensor, dim=(0, 2, 3))
        bank_var = torch.var(data_tensor, dim=(0, 2, 3))
        return bank_mean, bank_var

    # Euclidean distance between two (mean, var) descriptors
    def descriptor_distance(self, instance_descriptor, bank_descriptor):
        inst_mean, inst_var = instance_descriptor
        bank_mean, bank_var = bank_descriptor
        inst_concat = torch.cat([inst_mean, inst_var])
        bank_concat = torch.cat([bank_mean, bank_var])
        return torch.norm(inst_concat - bank_concat, p=2)

    def update_bank_descriptor(self, bank):
        bank["descriptor"] = self.compute_bank_descriptor(bank["items"])

    # Get adaptive threshold for deciding if a sample fits into a bank
    def get_dynamic_threshold(self, bank):
        bank_descriptor = bank["descriptor"]
        if bank_descriptor[0] is None or bank_descriptor[1] is None:
            return self.base_threshold
        std = torch.sqrt(bank_descriptor[1])
        avg_std = torch.mean(std).item()
        return self.base_threshold * (1 + avg_std)

    def merge_two_closest_banks(self):
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
            all_items.sort(key=lambda item: item.uncertainty)
            all_items = all_items[:self.capacity]

        new_class_items = [[] for _ in range(self.num_class)]
        for item in all_items:
            label = item.true_label
            if len(new_class_items[label]) < self.per_class:
                new_class_items[label].append(item)

        bank_i["items"] = new_class_items
        self.update_bank_descriptor(bank_i)

    def add_instance(self, instance):
        """
        Add a new sample to the memory bank system. Handles matching, insertion,
        replacement, and dynamic bank creation or merging if needed.
        """
        x, prediction, uncertainty, true_label = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0, true_label=true_label)
        instance_descriptor = self.compute_instance_descriptor(x)

        # Step 1: Find the closest matching bank
        best_bank = None
        best_distance = float('inf')
        for bank in self.banks:
            bank_descriptor = bank["descriptor"]
            if bank_descriptor[0] is None:
                continue
            d = self.descriptor_distance(instance_descriptor, bank_descriptor)
            if d < best_distance:
                best_distance = d
                best_bank = bank

        # Step 2: Create new bank or use best matching one
        if best_bank is None or (best_distance > self.get_dynamic_threshold(best_bank)):
            if len(self.banks) < self.mem_num:
                # Create a new bank directly
                new_bank = {
                    "items": [[] for _ in range(self.num_class)],
                    "descriptor": instance_descriptor
                }
                self.banks.append(new_bank)
                target_bank = new_bank
            else:
                # Memory full, consolidate before adding
                self.merge_two_closest_banks()
                new_bank = {
                    "items": [[] for _ in range(self.num_class)],
                    "descriptor": instance_descriptor
                }
                self.banks.append(new_bank)
                target_bank = new_bank
        else:
            target_bank = best_bank

        # Step 3: Score the item and attempt to insert
        class_idx = true_label
        new_score = self.heuristic_score(age=0, uncertainty=uncertainty, data=x, bank=target_bank)

        if self.remove_instance(target_bank, class_idx, new_score):
            target_bank["items"][class_idx].append(new_item)
            self.update_bank_descriptor(target_bank)

        # Step 4: Increment age of all samples in the target bank
        self.add_age(target_bank)


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
                s = self.heuristic_score(item.age, item.uncertainty, item.data, bank)
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

    def heuristic_score(self, age, uncertainty, data, bank):
        # Lower score means higher priority to be kept
        instance_descriptor = self.compute_instance_descriptor(data)
        bank_descriptor = bank["descriptor"]
        distance = self.descriptor_distance(instance_descriptor, bank_descriptor)
        score = self.lambda_t * (1 / (1 + math.exp(-age / self.capacity))) + \
                self.lambda_u * (uncertainty / math.log(self.num_class)) + \
                self.lambda_d * distance
        return score

    def get_memory(self, batch_mean, batch_var):
        """
        Retrieve a replay batch from memory:
        - 95% comes from the most similar memory bank to the current input batch descriptor
        - 5% comes from other banks (to promote diversity)
        - Returns a list of tensors (sup_data) and their corresponding ages (sup_age)
        """
        if not self.banks:
            return [], []  # No memory yet

        # Identify the closest bank to current input descriptor
        best_distance = float('inf')
        target_bank = None
        target_descriptor = (batch_mean, batch_var)

        for bank in self.banks:
            desc = bank["descriptor"]
            if desc[0] is None:
                continue  # Skip uninitialized banks
            d = self.descriptor_distance(target_descriptor, desc)
            if d < best_distance:
                best_distance, target_bank = d, bank

        if target_bank is None:
            return [], []  # Still no suitable bank found

        # Flatten all items from the selected bank into one list
        primary_items = [item for cls_items in target_bank["items"] for item in cls_items]
        total = len(primary_items)

        # Split into 95% primary, 5% secondary samples (at least 1 secondary)
        secondary_count = max(1, int(total * 0.5))
        primary_count = total - secondary_count

        # Sample primary items (either copy all or randomly pick a subset)
        selected_primary = (
            primary_items.copy()
            if primary_count >= total
            else random.sample(primary_items, primary_count)
        )

        # Collect all items from other banks (non-target bank)
        other_items = [
            item
            for bank in self.banks
            if bank is not target_bank
            for cls_items in bank["items"]
            for item in cls_items
        ]

        # Randomly sample secondary items
        if other_items:
            selected_secondary = (
                random.sample(other_items, secondary_count)
                if len(other_items) >= secondary_count
                else random.choices(other_items, k=secondary_count)
            )
        else:
            selected_secondary = []

        # Combine both and shuffle
        replay_items = selected_primary + selected_secondary
        random.shuffle(replay_items)

        # Return data and age lists
        sup_data = [item.data for item in replay_items]
        sup_age = [item.age for item in replay_items]
        return sup_data, sup_age

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

        # Step 1: Compute descriptor (mean, var) for each sample
        sample_descriptors = []
        for i in range(batch_samples.shape[0]):
            mean = torch.mean(batch_samples[i], dim=(1, 2))  # (C,)
            var = torch.var(batch_samples[i], dim=(1, 2))    # (C,)
            sample_descriptors.append((mean, var))

        # Step 2: Compute total distance to each bank
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

        # Step 3: Get top-K banks by lowest total distance
        sorted_ids = sorted(zip(valid_bank_ids, total_dists), key=lambda x: x[1])
        topk_bank_ids = [bank_id for bank_id, _ in sorted_ids[:topk]]

        # Step 4: Collect memory items from selected banks
        selected_items = []
        for bank_id in topk_bank_ids:
            for cls_items in self.banks[bank_id]["items"]:
                selected_items.extend(cls_items)

        # Step 5: Limit to max_samples with random shuffle
        random.shuffle(selected_items)
        selected_items = selected_items[:max_samples]

        # Step 6: Build outputs
        sup_data = [item.data for item in selected_items]
        sup_age = [item.age for item in selected_items]

        return sup_data, sup_age

