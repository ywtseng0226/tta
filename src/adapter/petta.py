import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from src.adapter.base_adapter import BaseAdapter
from src.utils.loss_func import self_training, softmax_entropy
from src.utils import set_named_submodule, get_named_submodule, petta_memory
from src.utils.custom_transforms import get_tta_transforms
from src.utils.bn_layers import RobustBN1d, RobustBN2d
from src.utils.petta_utils import split_up_model, get_source_loader

class PeTTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(PeTTA, self).__init__(cfg, model, optimizer)

        assert cfg.ADAPTER.PETTA.REGULARIZER in ["l2", "cosine", "none"]
        assert cfg.ADAPTER.PETTA.NORM_LAYER in ["rbn"]
        assert cfg.ADAPTER.PETTA.LOSS_FUNC in ["sce", "ce"]

        self.regularizer = cfg.ADAPTER.PETTA.REGULARIZER
        self.dataset_name = cfg.CORRUPTION.DATASET.replace("_recur", "")
        self.num_classes = cfg.CORRUPTION.NUM_CLASS
        self.alpha = cfg.ADAPTER.PETTA.ALPHA_0
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY

        # Model setup
        self.model_feat, self.model_clsf = split_up_model(model, cfg.MODEL.ARCH, self.dataset_name)
        self.model_ema = self.build_ema(model)
        self.model_ema_feat, self.model_ema_clsf = split_up_model(self.model_ema, cfg.MODEL.ARCH, self.dataset_name)
        self.model_init = self.build_ema(model).cuda()
        self.model_init_feat, self.model_init_clsf = split_up_model(self.model_init, cfg.MODEL.ARCH, self.dataset_name)
        self.init_model_state = self.model_init.state_dict()

        # Transformation
        self.transform = get_tta_transforms(cfg)

        # Feature statistics from source
        src_feat_mean, src_feat_cov = self.compute_source_features()

        # Memory modules
        self.sample_mem = petta_memory.PeTTAMemory(
            capacity=cfg.ADAPTER.RoTTA.MEMORY_SIZE,
            num_class=cfg.CORRUPTION.NUM_CLASS,
            lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T,
            lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U
        )
        self.proto_mem = petta_memory.PrototypeMemory(src_feat_mean, self.num_classes)
        self.divg_score = petta_memory.DivergenceScore(src_feat_mean, src_feat_cov)

        self.step = 0

    def compute_source_features(self, recompute=False):
        proto_dir = os.path.join(self.cfg.CKPT_DIR, "prototypes")
        os.makedirs(proto_dir, exist_ok=True)

        fname_base = f"protos_{self.dataset_name}_{self.cfg.MODEL.ARCH}_data_{self.cfg.ADAPTER.PETTA.PERCENTAGE}"
        fname_mean = os.path.join(proto_dir, f"{fname_base}_mean.pth")
        fname_cov = os.path.join(proto_dir, f"{fname_base}_cov.pth")

        if os.path.exists(fname_mean) and not recompute:
            print("Loading cached source features...")
            return torch.load(fname_mean).cuda(), torch.load(fname_cov).cuda()

        print("Extracting source prototypes...")
        _, src_loader = get_source_loader(
            dataset_name=self.dataset_name,
            root_dir=self.cfg.SRC_DATA_DIR,
            adaptation=self.cfg.ADAPTER.NAME,
            batch_size=512,
            ckpt_path=self.cfg.CKPT_PATH,
            percentage=self.cfg.ADAPTER.PETTA.PERCENTAGE,
            workers=min(self.cfg.ADAPTER.PETTA.NUM_WORKERS, os.cpu_count()),
            train_split=False
        )

        features, labels, gt_labels = [], [], []
        self.model.eval()
        with torch.no_grad():
            for x, y in tqdm.tqdm(src_loader):
                feat = self.model_feat(x.cuda())
                pred = self.model_clsf(feat).argmax(1).cpu()
                features.append(feat.view(feat.shape[:2]).cpu())
                labels.append(pred)
                gt_labels.append(y)
                if sum([len(f) for f in features]) > 100000:
                    break

        features = torch.cat(features)
        labels = torch.cat(labels)
        gt_labels = torch.cat(gt_labels)

        print("Pseudo-label acc:", (labels == gt_labels).float().mean().item())

        means, covs = [], []
        for i in range(self.num_classes):
            mask = labels == i
            means.append(features[mask].mean(0, keepdim=True))
            covs.append(torch.diagonal(features[mask].T.cov()).unsqueeze(0))

        means = torch.cat(means).cuda()
        covs = torch.cat(covs).cuda()
        torch.save(means, fname_mean)
        torch.save(covs, fname_cov)
        return means, covs

    def regularization_loss(self, model):
        loss = 0.0
        count = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            target = self.init_model_state[name].cuda()
            if self.regularizer == "l2":
                loss += ((param - target) ** 2).sum()
            elif self.regularizer == "cosine":
                loss += -F.cosine_similarity(param[None], target[None]).mean()
            count += 1
        return loss / count if count > 0 else torch.tensor(0.0).cuda()

    @staticmethod
    def update_ema_variables(ema_model, model, alpha):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - alpha) * ema_param.data[:] + alpha * param.data[:]
        return ema_model

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, label):
        self.step += 1

        # Generate pseudo-labels using EMA model
        with torch.no_grad():
            self.model_ema.eval()
            feat_ema = self.model_ema_feat(batch_data)
            logits_ema = self.model_ema_clsf(feat_ema)
            probs = torch.softmax(logits_ema, dim=1)
            pseudo_lbls = torch.argmax(probs, dim=1)
            entropy = torch.sum(-probs * torch.log(probs + 1e-6), dim=1)

        # Add to memory
        for i, data in enumerate(batch_data):
            self.sample_mem.add_instance((data, pseudo_lbls[i].item(), entropy[i].item(), label[i]))

        # Sample for adaptation
        sup_data, _ = self.sample_mem.get_memory()
        sup_data = torch.stack(sup_data)

        # Forward pass
        self.model.train()
        self.model_init.train()
        self.model_ema.train()

        ema_feat = self.model_ema_feat(sup_data)
        x_ema = self.model_ema_clsf(ema_feat)

        p_ori = self.model(sup_data)
        p_aug = self.model_clsf(self.model_feat(self.transform(sup_data)))

        init_feat = self.model_init_feat(sup_data)
        init_out = self.model_init_clsf(init_feat)

        # Losses
        if self.cfg.ADAPTER.PETTA.LOSS_FUNC == "sce":
            cls_loss = self_training(p_ori, p_aug, x_ema).mean()
        else:
            cls_loss = softmax_entropy(p_aug, x_ema).mean()

        reg_loss = self.regularization_loss(model)
        anchor_loss = softmax_entropy(p_aug, init_out).mean()
        reg_wgt = self.cfg.ADAPTER.PETTA.LAMBDA_0

        # Adaptively adjust weights
        if self.cfg.ADAPTER.PETTA.ADAPTIVE_LAMBDA or self.cfg.ADAPTER.PETTA.ADAPTIVE_ALPHA:
            uniq_lbls = torch.unique(pseudo_lbls)
            divergence = 1 - torch.exp(-self.divg_score(self.proto_mem.mem_proto[uniq_lbls], uniq_lbls))
            self.proto_mem.update(ema_feat.detach(), pseudo_lbls)
            if self.cfg.ADAPTER.PETTA.ADAPTIVE_LAMBDA:
                reg_wgt = divergence * reg_wgt
            if self.cfg.ADAPTER.PETTA.ADAPTIVE_ALPHA:
                self.alpha = (1 - divergence) * self.alpha

        # Backprop
        total_loss = cls_loss + reg_wgt * reg_loss + self.cfg.ADAPTER.PETTA.AL_WGT * anchor_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.alpha)
        return x_ema

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                NewBN = RobustBN1d if isinstance(sub_module, nn.BatchNorm1d) else RobustBN2d
                new_bn = NewBN(sub_module, self.cfg.ADAPTER.RoTTA.ALPHA)
                new_bn.requires_grad_(True)
                set_named_submodule(model, name, new_bn)
        return model
