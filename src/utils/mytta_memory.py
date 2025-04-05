import torch
import torch.nn.functional as F
import math
from copy import deepcopy
from torch import nn

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
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0, lambda_d=1.0, mem_num=5, eta=0.1, base_threshold=0.5):
        """
        :param capacity: 总容量（整个 memory 的最大样本数）
        :param num_class: 类别数
        :param lambda_t, lambda_u, lambda_d: age、uncertainty、distance 对启发式分数的权重
        :param mem_num: 最大 bank 数量（例如 5 个 domain）
        :param eta: 在线更新 bank 描述符的学习率（EMA 更新）
        :param base_threshold: 当 bank 内部方差信息不可用时的基础阈值
        """
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = capacity / num_class  # 每个类别最多存放的样本数
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.lambda_d = lambda_d
        self.mem_num = mem_num
        self.eta = eta
        self.base_threshold = base_threshold

        # banks 每个元素为一个字典：
        # {"items": [[], [], ...]  长度为 num_class，每个元素存储对应类别的 MemoryItem 列表
        #  "descriptor": (mean, var)} 作为该 bank 的聚类中心描述符
        self.banks = []

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
        """
        使用 bank["items"] 重新计算 bank 的描述符（聚类中心）。
        """
        bank["descriptor"] = self.compute_bank_descriptor(bank["items"])

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
        # 计算新实例描述符
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
            # 新 bank 的描述符直接初始化为该实例的描述符
            new_bank = {
                "items": [[] for _ in range(self.num_class)],
                "descriptor": instance_descriptor
            }
            self.banks.append(new_bank)
            target_bank = new_bank
            # print("add")
        else:
            target_bank = best_bank
            # 在线更新 bank 描述符（EMA 更新）
            old_mean, old_var = target_bank["descriptor"]
            new_mean = (1 - self.eta) * old_mean + self.eta * instance_descriptor[0]
            new_var = (1 - self.eta) * old_var + self.eta * instance_descriptor[1]
            target_bank["descriptor"] = (new_mean, new_var)

        # 将新实例加入 target_bank 对应类别
        class_idx = true_label  # 假设 true_label 为类别索引整数
        new_score = self.heuristic_score(age=0, uncertainty=uncertainty, data=x, bank=target_bank)
        if len(target_bank["items"][class_idx]) < self.per_class:
            target_bank["items"][class_idx].append(new_item)
        else:
            # 如果该类别已满，则尝试替换启发式分数较高的样本
            if self.remove_from_bank_for_class(target_bank, class_idx, new_score):
                target_bank["items"][class_idx].append(new_item)
                self.update_bank_descriptor(target_bank)
            else:
                # 否则丢弃该样本，或可调整策略
                pass

        self.add_age(target_bank)

    def remove_from_bank_for_class(self, bank, cls, new_score):
        """
        在 bank 指定类别列表中，找到启发式分数最高的样本，
        若其分数大于 new_score，则将其移除，返回 True；否则返回 False。
        """
        items = bank["items"][cls]
        if len(items) < self.per_class:
            return True
        max_score = None
        max_index = None
        for idx, item in enumerate(items):
            s = self.heuristic_score(item.age, item.uncertainty, item.data, bank)
            if max_score is None or s > max_score:
                max_score = s
                max_index = idx
        if max_score is not None and max_score > new_score:
            items.pop(max_index)
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
        根据传入的 (batch_mean, batch_var) 描述符返回一个 bank 内的数据。
        这里可采用简单的策略：返回第一个 bank的数据（或进一步选择最匹配的 bank）。
        返回格式：sup_data, sup_age
        """
        if not self.banks:
            return [], []
        # 例如：选择 descriptor 与 (batch_mean, batch_var) 距离最小的 bank
        target_bank = None
        best_distance = float('inf')
        target_descriptor = (batch_mean, batch_var)
        for bank in self.banks:
            if bank["descriptor"][0] is None:
                continue
            d = self.descriptor_distance(target_descriptor, bank["descriptor"])
            if d < best_distance:
                best_distance = d
                target_bank = bank
        # print("target_bank",target_bank)

        if target_bank is None:
            return [], []
        sup_data = []
        sup_age = []
        for class_items in target_bank["items"]:
            sup_data.extend([item.data for item in class_items])
            sup_age.extend([item.age for item in class_items])
        return sup_data, sup_age

