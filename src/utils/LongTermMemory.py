import torch
import torch.nn.functional as F
import math
import random

# MemoryItem: stores one sample and its meta info in memory bank
class MemoryItem:
    def __init__(self, data=None, label=None, domain=None):
        self.data = data
        self.label = label
        self.domain = domain 

class LongTermMemory:
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0, lambda_d=1.0,
                 max_bank_num=1, eta=0.1, base_threshold=0.3, ):
        self.capacity = capacity  # total memory capacity
        self.num_class = num_class # number of classes
        self.lambda_d = lambda_d  # distance factor
        self.memory = []  # memory bank
    
    def compute_instance_descriptor(self, data):
        data_mean = torch.mean(data, dim=(1, 2))
        data_var = torch.var(data, dim=(1, 2))
        return data_mean, data_var
    
    def descriptor_distance(self, inst_desc, ref_desc):
        # Euclidean distance between two descriptors
        inst_mean, inst_var = inst_desc
        ref_mean, ref_var = ref_desc
        inst_vec = torch.cat([inst_mean, inst_var])
        ref_vec = torch.cat([ref_mean, ref_var])
        return torch.norm(inst_vec - ref_vec, p=2)

    def heuristic_score(self, instance_descriptor):
        """
        Compute the *average* distance between the new instance and all items in memory.
        Larger distance (i.e., less similarity) means higher diversity, thus higher score.
        """
        if not self.memory:
            return float('inf')  # memory empty → always accept
        
        total_distance = 0.0
        count = 0
        for item in self.memory:
            ref_desc = self.compute_instance_descriptor(item.data)
            dist = self.descriptor_distance(instance_descriptor, ref_desc)
            total_distance += dist.item()
            count += 1
        avg_distance = total_distance / count
        return avg_distance  # The larger, the better

    def add_instance(self, data, label, domain):
        new_item = MemoryItem(data=data, label=label, domain=domain)
        instance_descriptor = self.compute_instance_descriptor(data)
        score = self.heuristic_score(instance_descriptor)

        if len(self.memory) < self.capacity:
            self.memory.append(new_item)
        else:
            # Replace the most similar (least diverse) item
            min_score = float('inf')
            min_idx = -1
            for idx, item in enumerate(self.memory):
                item_desc = self.compute_instance_descriptor(item.data)
                dist = self.descriptor_distance(instance_descriptor, item_desc).item()
                if dist < min_score:
                    min_score = dist
                    min_idx = idx

            if score > min_score:
                removed = self.memory.pop(min_idx)
                self.memory.append(new_item)
                return removed  # Return the replaced item (optional)

        return None
    
    def add_instance_v2(self, data, label, domain, desc_short_banks):
        """
        Add a new item to long-term memory. If capacity exceeded, remove the item
        most similar to short-term memory bank means.

        Args:
            data (Tensor): shape (C, H, W)
            label (int)
            domain (int)
            desc_short_banks (List[Tensor]): list of mean descriptors from short-term banks

        Returns:
            Optional[MemoryItem]: removed item, if any
        """
        new_item = MemoryItem(data=data, label=label, domain=domain)
        self.memory.append(new_item)

        if len(self.memory) <= self.capacity:
            return None

        # Compute average distance from each item to all short-term bank means
        item_scores = []
        for item in self.memory:
            item_mean, _ = self.compute_instance_descriptor(item.data)
            total_dist = 0.0
            for short_mean in desc_short_banks:
                total_dist += torch.norm(item_mean - short_mean, p=2).item()
            avg_dist = total_dist / len(desc_short_banks)
            item_scores.append(avg_dist)

        # Keep top-K farthest items
        sorted_indices = sorted(range(len(item_scores)), key=lambda i: -item_scores[i])
        keep_indices = set(sorted_indices[:self.capacity])
        remove_indices = set(range(len(self.memory))) - keep_indices

        if remove_indices:
            remove_idx = remove_indices.pop()
            removed_item = self.memory.pop(remove_idx)
            return removed_item
        else:
            return None



    def show_memory_status(self):
        print(f"[LongTermMemory] Total items: {len(self.memory)}")

        if not self.memory:
            return

        domain_counts = {}
        for item in self.memory:
            domain = item.domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        print(f"  Domain distribution:")
        for domain, count in sorted(domain_counts.items()):
            print(f"    Domain {domain}: {count}")

    def get_sup_data(self, short_sup_data, max_samples=16):
        """
        Select samples from long-term memory that are most dissimilar to short-term support data.

        Args:
            short_sup_data: list of tensors from short-term memory
            max_samples: number of samples to return

        Returns:
            sup_data: list of tensors
        """
        if not self.memory or not short_sup_data:
            return []

        # Compute descriptor for each short-term support item
        short_descriptors = [self.compute_instance_descriptor(x) for x in short_sup_data]

        # For each item in long-term memory, compute average distance to all short-term items
        scored_items = []
        for item in self.memory:
            item_desc = self.compute_instance_descriptor(item.data)
            total_dist = sum(self.descriptor_distance(item_desc, s_desc).item() for s_desc in short_descriptors)
            avg_dist = total_dist / len(short_descriptors)
            scored_items.append((avg_dist, item))

        # Sort by distance descending (most dissimilar first)
        scored_items.sort(key=lambda x: -x[0])

        # Select top-k
        selected = [item.data for _, item in scored_items[:max_samples]]
        return selected

