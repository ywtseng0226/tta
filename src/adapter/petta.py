import os
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
from src.adapter.base_adapter import BaseAdapter

from src.utils.loss_func import self_training, softmax_entropy
from src.utils import set_named_submodule, get_named_submodule, PeTTAMemory
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
        self.dataset_name = self.cfg.CORRUPTION.DATASET.replace("_recur", "")
        self.num_classes = self.cfg.CORRUPTION.NUM_CLASS

        self.transform = get_tta_transforms(cfg)
        self.alpha = cfg.ADAPTER.PETTA.ALPHA_0
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY

        self.model_feat, self.model_clsf = split_up_model(self.model, self.cfg.MODEL.ARCH, self.dataset_name)
        self.model_ema = self.build_ema(self.model)
        self.model_ema_feat, self.model_ema_clsf = split_up_model(self.model_ema, self.cfg.MODEL.ARCH, self.dataset_name)
        self.model_init = self.build_ema(self.model).cuda()
        self.model_init_feat, self.model_init_clsf = split_up_model(self.model_init, self.cfg.MODEL.ARCH, self.dataset_name)
        self.init_model_state = self.model_init.state_dict()

        src_feat_mean, src_feat_cov = self.compute_source_features()
        self.sample_mem = PeTTAMemory.PeTTAMemory(
            capacity=self.cfg.ADAPTER.RoTTA.MEMORY_SIZE,
            num_class=cfg.CORRUPTION.NUM_CLASS,
            lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T,
            lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U
        )
        self.proto_mem = PeTTAMemory.PrototypeMemory(src_feat_mean, self.num_classes)
        self.divg_score = PeTTAMemory.DivergenceScore(src_feat_mean, src_feat_cov)

        self.step = 0

    def compute_source_features(self, recompute=False):
        proto_dir_path = os.path.join(self.cfg.CKPT_DIR, "prototypes")

        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{self.cfg.CKPT_PATH.split(os.sep)[-1].split('_')[1]}_data_{self.cfg.ADAPTER.PETTA.PERCENTAGE}"
        else:
            fname = f"protos_{self.dataset_name}_{self.cfg.MODEL.ARCH}_data_{self.cfg.ADAPTER.PETTA.PERCENTAGE}"
        fname = os.path.join(proto_dir_path, fname)

        if os.path.exists(f'{fname}_mean.pth') and not recompute:
            print("Loading class-wise source features...")
            src_feat_mean = torch.load(f'{fname}_mean.pth')
            src_feat_cov = torch.load(f'{fname}_cov.pth')
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
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
            labels_gt_src = torch.tensor([])
            features_src = torch.tensor([])
            labels_src = torch.tensor([])

            self.model.cuda()
            self.model.eval()
            with torch.no_grad():
                for x, y_gt in tqdm.tqdm(src_loader):
                    tmp_features = self.model_feat(x.cuda())
                    y = self.model_clsf(tmp_features).argmax(1).cpu()
                    features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[:2]).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    labels_gt_src = torch.cat([labels_gt_src, y_gt], dim=0)
                    if len(features_src) > 100000:
                        break

            print("Pseudo label acc is: ", (labels_src == labels_gt_src).float().mean().item())
            src_feat_mean = torch.tensor([])
            src_feat_cov = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                src_feat_mean = torch.cat([src_feat_mean, features_src[mask].mean(dim=0, keepdim=True)], dim=0)
                src_feat_cov = torch.cat([src_feat_cov, torch.diagonal(features_src[mask].T.cov()).unsqueeze(0)], dim=0)

            torch.save(src_feat_mean, f'{fname}_mean.pth')
            torch.save(src_feat_cov, f'{fname}_cov.pth')

        return src_feat_mean.cuda(), src_feat_cov.cuda()

    def print_memory_status(self):
        all_ages = []
        class_counts = [len(class_items) for class_items in self.sample_mem.data]
        for class_items in self.sample_mem.data:
            all_ages.extend([item.age for item in class_items])
        if all_ages:
            print(f"[Memory Status @ Step {self.step}]")
            print(f"  Total samples: {len(all_ages)}")
            print(f"  Age → Min: {min(all_ages)}, Max: {max(all_ages)}, Avg: {sum(all_ages) / len(all_ages):.2f}")
            print(f"  Class sample counts (as list): {class_counts}")
        else:
            print(f"[Memory Status @ Step {self.step}] Empty memory.")

    def regularization_loss(self, model):
        reg_lss = 0.0
        count = 0
        if self.regularizer == "l2":
            for name, param in model.named_parameters():
                if param.requires_grad:
                    reg_lss += ((param - self.init_model_state[name].cuda()) ** 2).sum()
                    count += 1
            reg_lss /= count
        elif self.regularizer == "cosine":
            for name, param in model.named_parameters():
                if param.requires_grad:
                    reg_lss += -F.cosine_similarity(param[None], self.init_model_state[name][None].cuda()).mean()
                    count += 1
            reg_lss /= count
        elif self.regularizer == "none":
            reg_lss = torch.tensor([0.0], device="cuda")
        else:
            raise NotImplementedError()
        return reg_lss

    @staticmethod
    def update_ema_variables(ema_model, model, alpha):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - alpha) * ema_param.data[:] + alpha * param.data[:]
        return ema_model

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, label):
        self.step += 1
        with torch.no_grad():
            self.model_ema.eval()
            ema_sup_feat = self.model_ema_feat(batch_data)
            p_ema = self.model_ema_clsf(ema_sup_feat)
            predict = torch.softmax(p_ema, dim=1)
            pseudo_lbls = torch.argmax(predict, dim=1)
            entropy = torch.sum(-predict * torch.log(predict + 1e-6), dim=1)

        for i, data in enumerate(batch_data):
            self.sample_mem.add_instance((data, pseudo_lbls[i].item(), entropy[i].item(), label[i]))

        if self.step % self.update_frequency == 0:
            self.print_memory_status()

        sup_data, _ = self.sample_mem.get_memory()
        sup_data = torch.stack(sup_data)

        self.model.train()
        self.model_init.train()
        self.model_ema.train()

        ema_feat = self.model_ema_feat(sup_data)
        x_ema = self.model_ema_clsf(ema_feat)
        p_ori = self.model(sup_data)
        init_feat = self.model_init_feat(sup_data)
        init_model_out = self.model_init_clsf(init_feat)
        strong_sup_aug = self.transform(sup_data)
        stu_sup_feat = self.model_feat(strong_sup_aug)
        p_aug = self.model_clsf(stu_sup_feat)

        if self.cfg.ADAPTER.PETTA.LOSS_FUNC == "sce":
            cls_lss = self_training(x=p_ori, x_aug=p_aug, x_ema=x_ema).mean()
        else:
            cls_lss = softmax_entropy(p_aug, x_ema).mean()

        reg_lss = self.regularization_loss(model)
        anchor_lss = softmax_entropy(p_aug, init_model_out).mean()
        reg_wgt = self.cfg.ADAPTER.PETTA.LAMBDA_0

        if self.cfg.ADAPTER.PETTA.ADAPTIVE_LAMBDA or self.cfg.ADAPTER.PETTA.ADAPTIVE_ALPHA:
            lbl_uniq = torch.unique(pseudo_lbls)
            divg_scr = 1 - torch.exp(-self.divg_score(feats=self.proto_mem.mem_proto[lbl_uniq], pseudo_lbls=lbl_uniq))
            self.proto_mem.update(feats=ema_sup_feat.detach(), pseudo_lbls=pseudo_lbls)
            if self.cfg.ADAPTER.PETTA.ADAPTIVE_LAMBDA:
                reg_wgt = divg_scr * reg_wgt
            if self.cfg.ADAPTER.PETTA.ADAPTIVE_ALPHA:
                self.alpha = (1 - divg_scr) * self.cfg.ADAPTER.PETTA.ALPHA_0

        total_lss = cls_lss + reg_wgt * reg_lss + self.cfg.ADAPTER.PETTA.AL_WGT * anchor_lss
        optimizer.zero_grad()
        total_lss.backward()
        optimizer.step()

        self.update_ema_variables(self.model_ema, self.model, self.alpha)
        return p_ema

    def configure_model(self, model: nn.Module):
        model.requires_grad_(False)
        normlayer_names = []
        for name, sub_module in model.named_modules():
            if isinstance(sub_module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                normlayer_names.append(name)
        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            NewBN = RobustBN1d if isinstance(bn_layer, nn.BatchNorm1d) else RobustBN2d
            momentum_bn = NewBN(bn_layer, self.cfg.ADAPTER.RoTTA.ALPHA)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model
