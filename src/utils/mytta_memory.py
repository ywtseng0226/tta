import torch
import torch.nn.functional as F
import math
from copy import deepcopy
from torch import nn
import random

def compute_feat_mean(feats, pseudo_lbls):
    lbl_uniq = torch.unique(pseudo_lbls)
    lbl_group = [torch.where(pseudo_lbls==l)[0] for l in lbl_uniq]
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
            return ((input-target).pow(2) / (target_cov + 1e-6)).mean()
        self.lss =  GSSLoss
    
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
        lbl_group = [torch.where(pseudo_lbls==l)[0] for l in lbl_uniq]
        for i, lbl_idcs in enumerate(lbl_group):
            psd_lbl = lbl_uniq[i]
            batch_avg =  feats[lbl_idcs].mean(axis=0)
            self.mem_proto[psd_lbl] = (1-nu) * self.mem_proto[psd_lbl] + nu * batch_avg

    def get_mem_prototype(self):
        return self.mem_proto


class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0, true_label=None):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age
        self.true_label = true_label

    def increase_age(self):
        if not self.empty():
            self.age += 1
            # print('age',self.age)

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"
    
# Adapted from: https://github.com/BIT-DA/RoTTA/blob/main/core/utils/memory.py
class MyTTAMemory:
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
        # repulsion params
        self.repulse_eta0 = repulse_eta0

        self.banks = []

    def repulse_banks(self, eps=1e-6):
        # 少于两个 bank，没得比就跳过
        if len(self.banks) < 2:
            return

        # 1) 计算所有两两距离，得 d_min 和 d_avg
        dists = []
        for i in range(len(self.banks)):
            m_i, _ = self.banks[i]["descriptor"]
            for j in range(i+1, len(self.banks)):
                m_j, _ = self.banks[j]["descriptor"]
                d = torch.norm(m_i - m_j).item()
                dists.append(d)
        d_min = min(dists)
        d_avg = sum(dists) / len(dists)

        # 2) 用 d_avg 作为动态目标距离
        eta = self.repulse_eta0 * (d_avg / (d_min + eps))
        # 可选：给步长做上下界
        eta = max(min(eta, self.repulse_eta0 * 10), self.repulse_eta0 * 0.1)

        # 3) 打印调试信息，观察收敛情况
        # print(f"[Repulse] banks={len(self.banks)}  d_min={d_min:.4f}  d_avg={d_avg:.4f}  eta={eta:.6f}")

        # 4) 真正做排斥更新
        for i, bi in enumerate(self.banks):
            mean_i, var_i = bi["descriptor"]
            delta = torch.zeros_like(mean_i)
            for j, bj in enumerate(self.banks):
                if i == j:
                    continue
                mean_j, _ = bj["descriptor"]
                diff = mean_i - mean_j
                delta += eta * diff / (diff.norm()**2 + eps)
            bi["descriptor"] = (mean_i + delta, var_i)



    def compute_instance_descriptor(self, data):
        """
        计算单个图片的描述符。
        假设 data 为形状 (C, H, W) 的 tensor，
        则按通道计算均值和方差，返回 (mean, var) 都为形状 (C,) 的 tensor。
        """
        data_mean = torch.mean(data, dim=(1, 2))
        data_var = torch.var(data, dim=(1, 2))
        # print(-1)
        # print(len(self.banks))
        return data_mean, data_var

    def compute_bank_descriptor(self, bank_items):
        """
        计算 bank 的描述符：将 bank_items 中所有 MemoryItem.data 拼接，
        在样本、H 和 W 维度上计算均值和方差，返回 (mean, var) 形状均为 (C,) 的 tensor。
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
        计算实例描述符与 bank 描述符之间的距离。
        两者均为 (mean, var)，先拼接为 2C 维向量，再计算欧氏距离。
        """
        inst_mean, inst_var = instance_descriptor
        bank_mean, bank_var = bank_descriptor
        inst_concat = torch.cat([inst_mean, inst_var])
        bank_concat = torch.cat([bank_mean, bank_var])
        return torch.norm(inst_concat - bank_concat, p=2)

    def update_bank_descriptor(self, bank):
        bank["descriptor"] = self.compute_bank_descriptor(bank["items"])
        # 全局排斥一下
        # self.repulse_banks()


    def get_dynamic_threshold(self, bank):
        """
        根据 bank 的描述符（特别是方差部分）计算动态阈值。
        例如：阈值 = base_threshold * (1 + average_std)
        如果 bank 内没有有效信息，则返回 base_threshold。
        """
        bank_descriptor = bank["descriptor"]
        if bank_descriptor[0] is None or bank_descriptor[1] is None:
            return self.base_threshold
        std = torch.sqrt(bank_descriptor[1])
        avg_std = torch.mean(std).item()
        return self.base_threshold * (1 + avg_std)

    def add_instance(self, instance):
        """
        使用在线 k-means 思想添加新实例到 memory 中。
        instance 格式：(x, prediction, uncertainty, true_label)
        """
        x, prediction, uncertainty, true_label = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0, true_label=true_label)
        # 计算新实例的描述符
        instance_descriptor = self.compute_instance_descriptor(x)
        # 遍历已有 banks，找出距离最近的 bank
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

        # 如果没有 bank 或者（距离超过动态阈值且 bank 数量未满）则新建 bank
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
        # print(f"[MyTTAMemory] adding to bank #{bank_idx}")

        # 将新实例加入 target_bank 对应类别
        class_idx = true_label  # 假设 true_label 为类别索引整数
        new_score = self.heuristic_score(age=0, uncertainty=uncertainty, data=x, bank=target_bank)

        if self.remove_instance(target_bank, class_idx, new_score):
            target_bank["items"][class_idx].append(new_item)
            self.update_bank_descriptor(target_bank)
        else:
            # 否则丢弃该样本，或可根据需求调整策略
            pass

        self.add_age(target_bank)


    def remove_instance(self, bank, cls, new_score):

        items = bank["items"][cls]
        class_occupied = len(items)
        all_occupancy = sum(len(lst) for lst in bank["items"])
        # print("all_occupancy:",all_occupancy)
        # 如果目标类别未满且整体容量未达到上限，直接允许添加
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                max_count = max(len(lst) for lst in bank["items"])
                majority_classes = [i for i, lst in enumerate(bank["items"]) if len(lst) == max_count]
                return self.remove_from_classes(bank,majority_classes,new_score,)
        else:
            # print("remove")
            return self.remove_from_classes(bank,[cls],new_score)
        
    def remove_from_classes(self, bank, classes, score_base):
        """
        在 bank 指定的多个类别中，尝试删除启发式分数最高且大于 score_base 的样本。
        
        参数：
        bank：包含内存样本的 bank 字典，其结构为 {"items": [[], [], ...], "descriptor": ...}
        classes：待检查的类别列表（例如 [cls] 或 majority_classes）
        score_base：新样本的启发式分数，只有删除的样本分数大于此值时才执行删除
        
        返回：
        如果成功删除一个样本则返回 True，否则返回 False。
        """
        max_score = None
        max_class = None
        max_index = None
        
        # 遍历待检查的类别
        for c in classes:
            items = bank["items"][c]
            for idx, item in enumerate(items):
                # 计算每个样本的启发式分数（age, uncertainty, data，bank 信息会影响评分）
                s = self.heuristic_score(item.age, item.uncertainty, item.data, bank)
                if max_score is None or s > max_score:
                    max_score = s
                    max_class = c
                    max_index = idx

        # 如果找到的最大分数大于 score_base，则删除该样本并返回 True
        if max_class is not None and max_score > score_base:
            d = bank["items"][max_class].pop(max_index)
            del d
            return True
        else:
            return False


    def add_age(self, bank):
        """
        对 bank 内所有样本的 age 加 1
        """
        for class_items in bank["items"]:
            for item in class_items:
                item.increase_age()

    def heuristic_score(self, age, uncertainty, data, bank):
        """
        计算启发式分数：
          score = λ_t * (1/(1+exp(-age/capacity))) + λ_u * (uncertainty/ log(num_class)) + λ_d * (distance)
        其中 distance 为新实例描述符与 bank 描述符之间的距离（越小越好）。
        分数越低，样本越有资格加入 bank。
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
        建立一个“重放”bank：95%来自最匹配的bank，5%随机抽自其他banks，
        总样本数与匹配bank一致，并返回 data 和 age 列表。
        """
        if not self.banks:
            return [], []

        # 找到最匹配的 bank
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

        # 把匹配bank里所有item平铺
        primary_items = [item
                         for cls_items in target_bank["items"]
                         for item in cls_items]
        total = len(primary_items)
        if total == 0:
            return [], []

        # 95% 主样本，5% 次样本
        secondary_count = max(1, int(total * 0.5))
        primary_count   = total - secondary_count

  

        # 随机选主样本
        if primary_count >= total:
            selected_primary = primary_items.copy()
        else:
            selected_primary = random.sample(primary_items, primary_count)

        # 收集其他banks所有item，再随机抽次样本
        other_items = [item
                       for bank in self.banks if bank is not target_bank
                       for cls_items in bank["items"]
                       for item in cls_items]
        if other_items:
            if len(other_items) >= secondary_count:
                selected_secondary = random.sample(other_items, secondary_count)
            else:
                selected_secondary = random.choices(other_items, k=secondary_count)
        else:
            selected_secondary = []

        # 合并、打乱顺序
        replay_items = selected_primary + selected_secondary
        random.shuffle(replay_items)

        # 拆出 data 和 age
        sup_data = [item.data for item in replay_items]
        sup_age  = [item.age  for item in replay_items]
        return sup_data, sup_age
