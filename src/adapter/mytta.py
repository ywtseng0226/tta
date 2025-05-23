import os
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from src.adapter.base_adapter import BaseAdapter
from src.utils.loss_func import self_training, softmax_entropy
from src.utils import set_named_submodule, get_named_submodule
from src.utils import PeTTAMemory, ShortTermMemory, LongTermMemory
from src.utils.custom_transforms import get_tta_transforms
from src.utils.bn_layers import RobustBN1d, RobustBN2d
from src.utils.petta_utils import split_up_model, get_source_loader

class MyTTA(BaseAdapter):
    def __init__(self, cfg, model, optimizer):
        super(MyTTA, self).__init__(cfg, model, optimizer)

        assert cfg.ADAPTER.PETTA.REGULARIZER in ["l2", "cosine", "none"]
        assert cfg.ADAPTER.PETTA.NORM_LAYER in ["rbn"]
        assert cfg.ADAPTER.PETTA.LOSS_FUNC in ["sce", "ce"]

        self.regularizer = cfg.ADAPTER.PETTA.REGULARIZER
        self.dataset_name = self.cfg.CORRUPTION.DATASET.replace("_recur", "")
        self.num_classes = self.cfg.CORRUPTION.NUM_CLASS

        # Mean teacher parameters
        self.transform = get_tta_transforms(cfg)
        self.alpha = cfg.ADAPTER.PETTA.ALPHA_0
        self.update_frequency = cfg.ADAPTER.RoTTA.UPDATE_FREQUENCY

        # Split model into feature extractor and classifier
        self.model_feat, self.model_clsf = split_up_model(self.model, self.cfg.MODEL.ARCH, self.dataset_name)

        # Build EMA teacher model
        self.model_ema = self.build_ema(self.model)
        self.model_ema_feat, self.model_ema_clsf = split_up_model(self.model_ema, self.cfg.MODEL.ARCH, self.dataset_name)

        # Initialize source model snapshot (for regularization)
        self.model_init = self.build_ema(self.model).cuda()
        self.model_init_feat, self.model_init_clsf = split_up_model(self.model_init, self.cfg.MODEL.ARCH, self.dataset_name)
        self.init_model_state = self.model_init.state_dict()

        # Compute source-domain feature statistics
        src_feat_mean, src_feat_cov = self.compute_source_features()

        # Initialize memory modules
        self.proto_mem = PeTTAMemory.PrototypeMemory(src_feat_mean, self.num_classes)
        self.divg_score = PeTTAMemory.DivergenceScore(src_feat_mean, src_feat_cov)
        self.short_term_memory = ShortTermMemory.ShortTermMemory(
            capacity=self.cfg.ADAPTER.MYTTA.STMEM_CAPACITY,
            num_class=cfg.CORRUPTION.NUM_CLASS,
            lambda_t=cfg.ADAPTER.RoTTA.LAMBDA_T,
            lambda_u=cfg.ADAPTER.RoTTA.LAMBDA_U,
            max_bank_num=cfg.ADAPTER.MYTTA.STMEM_MAX_CLUS, 
            base_threshold=cfg.ADAPTER.MYTTA.BASE_THRESHOLD,
        )
        self.long_term_memory = LongTermMemory.LongTermMemory(
            capacity=self.cfg.ADAPTER.MYTTA.LTMEM_CAPACITY,
            num_class=cfg.CORRUPTION.NUM_CLASS,
        )
        # Initialize step counter
        self.step = 0

    def compute_source_features(self, recompute=False):
        # Load cached prototypes if available
        proto_dir_path = os.path.join(self.cfg.CKPT_DIR, "prototypes")
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{self.cfg.CKPT_PATH.split(os.sep)[-1].split('_')[1]}_data_{self.cfg.ADAPTER.PETTA.PERCENTAGE}"
        else:
            fname = f"protos_{self.dataset_name}_{self.cfg.MODEL.ARCH}_data_{self.cfg.ADAPTER.PETTA.PERCENTAGE}"
        fname = os.path.join(proto_dir_path, fname)

        if os.path.exists(f'{fname}_mean.pth') and not recompute:
            print("Loading cached source prototypes...")
            src_feat_mean = torch.load(f'{fname}_mean.pth')
            src_feat_cov = torch.load(f'{fname}_cov.pth')
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            print("Computing source prototypes...")
            _, src_loader = get_source_loader(
                dataset_name=self.dataset_name,
                root_dir=self.cfg.SRC_DATA_DIR,
                adaptation=self.cfg.ADAPTER.NAME,
                batch_size=512,
                ckpt_path=self.cfg.CKPT_PATH,
                percentage=self.cfg.ADAPTER.PETTA.PERCENTAGE,
                workers=min(self.cfg.ADAPTER.PETTA.NUM_WORKERS, os.cpu_count()),
                train_split=False  # use val/test split for statistics
            )

            labels_gt_src = torch.tensor([])
            features_src = torch.tensor([])
            labels_src = torch.tensor([])

            self.model.cuda()
            self.model.eval()
            with torch.no_grad():
                for dat in tqdm.tqdm(src_loader):
                    x, y_gt = dat[0], dat[1]
                    tmp_features = self.model_feat(x.cuda())
                    y = self.model_clsf(tmp_features).argmax(1).cpu()
                    features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[:2]).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    labels_gt_src = torch.cat([labels_gt_src, y_gt], dim=0)
                    if len(features_src) > 100000:
                        break

            print("Pseudo label accuracy on source: ", (labels_src == labels_gt_src).float().mean().item())

            src_feat_mean = torch.tensor([])
            src_feat_cov = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                src_feat_mean = torch.cat([src_feat_mean, features_src[mask].mean(dim=0, keepdim=True)], dim=0)
                src_feat_cov = torch.cat([src_feat_cov, torch.diagonal(features_src[mask].T.cov()).unsqueeze(0)], dim=0)

            torch.save(src_feat_mean, f'{fname}_mean.pth')
            torch.save(src_feat_cov, f'{fname}_cov.pth')

        return src_feat_mean.cuda(), src_feat_cov.cuda()

    def regularization_loss(self, model):
        reg_lss, count = 0.0, 0
        if self.regularizer == "l2":
            for name, param in model.named_parameters():
                if param.requires_grad:
                    reg_lss += ((param - self.init_model_state[name].cuda())**2).sum()
                    count += 1
            reg_lss /= count
        elif self.regularizer == "cosine":
            for name, param in model.named_parameters():
                if param.requires_grad:
                    reg_lss += -F.cosine_similarity(param[None, ...], self.init_model_state[name][None, ...].cuda()).mean()
                    count += 1
            reg_lss /= count
        elif self.regularizer == "none":
            reg_lss = torch.tensor([0.0], device="cuda")
        else:
            raise NotImplementedError()
        return reg_lss

    @staticmethod
    def update_ema_variables(ema_model, model, alpha):
        # EMA parameter update (Equation 3)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - alpha) * ema_param.data[:] + alpha * param.data[:]
        return ema_model

    def visualize_support_data(self, short_data, long_data, step):
        """
        Visualize support data from short-term and long-term memory using PCA.

        Args:
            short_data (list[Tensor])
            long_data (list[Tensor])
            step (int)
        """
        if not short_data and not long_data:
            return

        all_feats = []
        color_labels = []

        for data in short_data:
            feat = torch.mean(data, dim=(1, 2)).cpu().numpy()  # (C,)
            all_feats.append(feat)
            color_labels.append(0)  # 0 = short-term

        for data in long_data:
            feat = torch.mean(data, dim=(1, 2)).cpu().numpy()  # (C,)
            all_feats.append(feat)
            color_labels.append(1)  # 1 = long-term

        features = np.stack(all_feats)
        color_labels = np.array(color_labels)

        # PCA to 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        plt.figure(figsize=(7, 6))

        # Plot each group separately for correct legend
        short_mask = color_labels == 0
        long_mask = color_labels == 1
        plt.scatter(features_2d[short_mask, 0], features_2d[short_mask, 1],
                    c='tab:blue', label="Short-Term", alpha=0.7)
        plt.scatter(features_2d[long_mask, 0], features_2d[long_mask, 1],
                    c='tab:red', label="Long-Term", alpha=0.7)

        plt.legend(loc="best")
        plt.title(f"Support Data Visualization @ Step {step}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.xlim(-1.0, 1.0)
        plt.ylim(-0.5, 0.5)
        plt.tight_layout()

        vis_dir = os.path.join(self.cfg.OUTPUT_DIR, "support_vis")
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, f"support_vis_step_{step:05d}.png")
        plt.savefig(vis_path)
        plt.close()


    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer, label):
        self.step += 1

        # Generate pseudo labels using EMA teacher
        with torch.no_grad():
            self.model_ema.eval()
            ema_sup_feat = self.model_ema_feat(batch_data)
            p_ema = self.model_ema_clsf(ema_sup_feat)
            predict = torch.softmax(p_ema, dim=1)
            pseudo_lbls = torch.argmax(predict, dim=1)
            entropy = torch.sum(-predict * torch.log(predict + 1e-6), dim=1)

        data_tensor = batch_data if isinstance(batch_data, torch.Tensor) else torch.stack(batch_data)

        # Add each sample to memory as before
        for i, data in enumerate(batch_data):
            removed = self.short_term_memory.add_instance((data, pseudo_lbls[i].item(), entropy[i].item(), label['label'][i].item(), label['domain'][i].item()))
            # if removed: 
            #     self.long_term_memory.add_instance(removed.data, removed.label, removed.domain)
        
        # self.short_term_memory.show_memory_status()
        # self.long_term_memory.show_memory_status()

        # Get the support data from shor-term memory
        sup_data_short = self.short_term_memory.get_sup_data(data_tensor, topk=self.cfg.ADAPTER.MYTTA.STMEM_TOPK_CLUS) 
        # sup_data_long = self.long_term_memory.get_sup_data(sup_data_short)
        # self.visualize_support_data(sup_data_short, sup_data_long, self.step)
        sup_data = sup_data_short

        # Convert to tensors if needed
        sup_data = torch.stack(sup_data)

        # Get predictions from student and teacher models
        self.model_ema.train()
        ema_feat = self.model_ema_feat(sup_data)
        x_ema = self.model_ema_clsf(ema_feat)

        self.model.train()
        self.model_init.train()
        p_ori = self.model(sup_data)

        init_feat = self.model_init_feat(sup_data)
        init_model_out = self.model_init_clsf(init_feat)

        strong_sup_aug = self.transform(sup_data)
        stu_sup_feat = self.model_feat(strong_sup_aug)
        p_aug = self.model_clsf(stu_sup_feat)

        # Classification loss
        if self.cfg.ADAPTER.PETTA.LOSS_FUNC == "sce":
            cls_lss = self_training(x=p_ori, x_aug=p_aug, x_ema=x_ema).mean()
        else:
            cls_lss = softmax_entropy(p_aug, x_ema).mean()

        reg_lss = self.regularization_loss(model)
        anchor_lss = softmax_entropy(p_aug, init_model_out).mean()
        reg_wgt = self.cfg.ADAPTER.PETTA.LAMBDA_0

        # Adaptive update of lambda and alpha
        if self.cfg.ADAPTER.PETTA.ADAPTIVE_LAMBDA or self.cfg.ADAPTER.PETTA.ADAPTIVE_ALPHA:
            lbl_uniq = torch.unique(pseudo_lbls)
            divg_scr = 1 - torch.exp(-self.divg_score(self.proto_mem.mem_proto[lbl_uniq], pseudo_lbls=lbl_uniq))
            self.proto_mem.update(feats=ema_sup_feat.detach(), pseudo_lbls=pseudo_lbls)
            if self.cfg.ADAPTER.PETTA.ADAPTIVE_LAMBDA:
                reg_wgt = divg_scr * self.cfg.ADAPTER.PETTA.LAMBDA_0
            if self.cfg.ADAPTER.PETTA.ADAPTIVE_ALPHA:
                self.alpha = (1 - divg_scr) * self.cfg.ADAPTER.PETTA.ALPHA_0

        # Final loss and update
        total_lss = cls_lss + reg_wgt * reg_lss + self.cfg.ADAPTER.PETTA.AL_WGT * anchor_lss
        optimizer.zero_grad()
        total_lss.backward()
        optimizer.step()

        # Update EMA teacher
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
