import torch
import torch.nn.functional as F
import math
from copy import deepcopy
from torch import nn
import random

def compute_feat_mean(feats, pseudo_lbls):
    lbl_uniq = torch.unique(pseudo_lbls)
    lbl_group = [torch.where(pseudo_lbls == l)[0] for l in lbl_uniq]
    group_avgs = []
    for i, lbl_idcs in enumerate(lbl_group):
        group_avgs.append(feats[lbl_idcs].mean(axis=0).unsqueeze(0))
    return lbl_uniq, group_avgs 

class DivergenceScore(nn.Module):
    def __init__(self, src_prototype, src_prototype_cov):
        super().__init__()
        self.src_proto = src_prototype
        self.src_proto_cov = src_prototype_cov

        def GSSLoss(input, target, target_cov):
            return ((input - target).pow(2) / (target_cov + 1e-6)).mean()

        self.lss = GSSLoss

    def forward(self, feats, pseudo_lbls):
        lbl_uniq, group_avgs = compute_feat_mean(feats, pseudo_lbls)
        return self.lss(
            torch.cat(group_avgs, dim=0),
            self.src_proto[lbl_uniq],
            self.src_proto_cov[lbl_uniq],
        )

class PrototypeMemory:
    def __init__(self, src_prototype, num_classes) -> None:
        self.src_proto = src_prototype.squeeze(1)
        self.mem_proto = deepcopy(self.src_proto)
        self.num_classes = num_classes
        self.src_proto_l2 = torch.cdist(self.src_proto, self.src_proto, p=2)

    def update(self, feats, pseudo_lbls, nu=0.05):
        lbl_uniq = torch.unique(pseudo_lbls)
        lbl_group = [torch.where(pseudo_lbls == l)[0] for l in lbl_uniq]
        for i, lbl_idcs in enumerate(lbl_group):
            psd_lbl = lbl_uniq[i]
            batch_avg = feats[lbl_idcs].mean(axis=0)
            self.mem_proto[psd_lbl] = (1 - nu) * self.mem_proto[psd_lbl] + nu * batch_avg

    def get_mem_prototype(self):
        return self.mem_proto

class MemoryItem:
    """
    Represents a single memory item storing a sample, its uncertainty, age, and ground-truth label.
    """
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

class MyTTAMemory:
    """
    A memory module designed for online test-time adaptation.
    It stores class-wise samples and organizes them into dynamic clusters (banks) for better selection.
    """
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0, lambda_d=1.0,
                 mem_num=5, eta=0.1, base_threshold=0.5,
                 repulse_eta0=0.1):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = capacity / num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.lambda_d = lambda_d
        self.mem_num = mem_num
        self.eta = eta
        self.base_threshold = base_threshold
        self.repulse_eta0 = repulse_eta0
        self.banks = []

    def compute_instance_descriptor(self, data):
        """
        Compute descriptor (mean, var) of a single image tensor with shape (C, H, W).
        """
        data_mean = torch.mean(data, dim=(1, 2))
        data_var = torch.var(data, dim=(1, 2))
        return data_mean, data_var

    def compute_bank_descriptor(self, bank_items):
        """
        Compute descriptor of a bank by aggregating all MemoryItems' data.
        Returns (mean, var) over shape (C,).
        """
        all_items = []
        for class_items in bank_items:
            all_items.extend(class_items)
        if len(all_items) == 0:
            return None, None
        data_tensor = torch.stack([item.data for item in all_items])
        bank_mean = torch.mean(data_tensor, dim=(0, 2, 3))
        bank_var = torch.var(data_tensor, dim=(0, 2, 3))
        return bank_mean, bank_var

    def descriptor_distance(self, instance_descriptor, bank_descriptor):
        """
        Compute Euclidean distance between instance and bank descriptors.
        Descriptors are both (mean, var) tuples.
        """
        inst_mean, inst_var = instance_descriptor
        bank_mean, bank_var = bank_descriptor
        inst_concat = torch.cat([inst_mean, inst_var])
        bank_concat = torch.cat([bank_mean, bank_var])
        return torch.norm(inst_concat - bank_concat, p=2)

    def update_bank_descriptor(self, bank):
        """Update descriptor for the given bank."""
        bank["descriptor"] = self.compute_bank_descriptor(bank["items"])

    def get_dynamic_threshold(self, bank):
        """
        Dynamically adjust threshold based on variance in bank descriptor.
        """
        bank_descriptor = bank["descriptor"]
        if bank_descriptor[0] is None or bank_descriptor[1] is None:
            return self.base_threshold
        std = torch.sqrt(bank_descriptor[1])
        avg_std = torch.mean(std).item()
        return self.base_threshold * (1 + avg_std)

    def add_instance(self, instance):
        """
        Add new instance to memory using online k-means-like clustering strategy.
        """
        x, prediction, uncertainty, true_label = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0, true_label=true_label)
        instance_descriptor = self.compute_instance_descriptor(x)

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

        if best_bank is None or (best_distance > self.get_dynamic_threshold(best_bank) and len(self.banks) < self.mem_num):
            new_bank = {
                "items": [[] for _ in range(self.num_class)],
                "descriptor": instance_descriptor
            }
            self.banks.append(new_bank)
            target_bank = new_bank
        else:
            target_bank = best_bank

        bank_idx = self.banks.index(target_bank)
        class_idx = true_label
        new_score = self.heuristic_score(age=0, uncertainty=uncertainty, data=x, bank=target_bank)

        if self.remove_instance(target_bank, class_idx, new_score):
            target_bank["items"][class_idx].append(new_item)
            self.update_bank_descriptor(target_bank)

        self.add_age(target_bank)

    def remove_instance(self, bank, cls, new_score):
        """
        Remove a memory item if the class is full or total capacity is reached.
        Replacement is decided based on heuristic score comparison.
        """
        items = bank["items"][cls]
        class_occupied = len(items)
        all_occupancy = sum(len(lst) for lst in bank["items"])

        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                max_count = max(len(lst) for lst in bank["items"])
                majority_classes = [i for i, lst in enumerate(bank["items"]) if len(lst) == max_count]
                return self.remove_from_classes(bank, majority_classes, new_score)
        else:
            return self.remove_from_classes(bank, [cls], new_score)

    def remove_from_classes(self, bank, classes, score_base):
        """
        Remove item from specified class if its heuristic score > new instance's score.
        """
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
        """
        Increment age of all memory items within the given bank.
        """
        for class_items in bank["items"]:
            for item in class_items:
                item.increase_age()

    def heuristic_score(self, age, uncertainty, data, bank):
        """
        Compute a heuristic score based on age, uncertainty, and descriptor distance.
        Lower score indicates higher priority for retention.
        """
        instance_descriptor = self.compute_instance_descriptor(data)
        bank_descriptor = bank["descriptor"]
        distance = self.descriptor_distance(instance_descriptor, bank_descriptor)
        score = self.lambda_t * (1 / (1 + math.exp(-age / self.capacity))) + \
                self.lambda_u * (uncertainty / math.log(self.num_class)) + \
                self.lambda_d * distance
        return score

    def get_memory(self, batch_mean, batch_var):
        """
        Retrieve a replay batch: 95% from best matching bank, 5% from others.
        Return: (sample list, age list)
        """
        if not self.banks:
            return [], []

        best_distance = float('inf')
        target_bank = None
        target_descriptor = (batch_mean, batch_var)
        for bank in self.banks:
            desc = bank["descriptor"]
            if desc[0] is None: 
                continue
            d = self.descriptor_distance(target_descriptor, desc)
            if d < best_distance:
                best_distance, target_bank = d, bank

        if target_bank is None:
            return [], []

        primary_items = [item for cls_items in target_bank["items"] for item in cls_items]
        total = len(primary_items)
        if total == 0:
            return [], []

        secondary_count = max(1, int(total * 0.5))
        primary_count = total - secondary_count

        selected_primary = primary_items.copy() if primary_count >= total else random.sample(primary_items, primary_count)

        other_items = [item for bank in self.banks if bank is not target_bank for cls_items in bank["items"] for item in cls_items]
        if other_items:
            selected_secondary = random.sample(other_items, secondary_count) if len(other_items) >= secondary_count else random.choices(other_items, k=secondary_count)
        else:
            selected_secondary = []

        replay_items = selected_primary + selected_secondary
        random.shuffle(replay_items)

        sup_data = [item.data for item in replay_items]
        sup_age = [item.age for item in replay_items]
        return sup_data, sup_age
