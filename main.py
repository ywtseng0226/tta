import torch
import argparse
from src.configs import cfg
from src.model import build_model
from src.data import build_loader
from src.optim import build_optimizer
from src.adapter import build_adapter
from tqdm import tqdm
from setproctitle import setproctitle
import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
import torch.multiprocessing
import pandas as pd
from src.utils import set_random_seed

def plot_error_trend(error_rates, output_dir, filename="tta_error_trend.png"):
    """
    Plot batch-wise error rates and save as a figure.

    Args:
        error_rates (list[float]): A list of error rate values (one per batch).
        output_dir (str): Directory to save the output figure.
        filename (str): Name of the figure file (default: "tta_error_trend.png").
    """
    if not error_rates:
        print("No error rate data to plot.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(error_rates, color='tab:red', linewidth=1.5)
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.title("TTA Batch-wise Error Rate Trend", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Saved error trend plot to: {save_path}")

def save_error_trend_to_csv(error_rates, output_dir, filename="tta_error_trend.csv"):
    """
    Save batch-wise error rates to a CSV file.

    Args:
        error_rates (list[float]): A list of error rate values (one per batch).
        output_dir (str): Directory to save the CSV file.
        filename (str): Name of the CSV file (default: "tta_error_trend.csv").
    """
    df = pd.DataFrame({
        "Batch Index": list(range(len(error_rates))),
        "Error Rate": error_rates
    })
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    df.to_csv(save_path, index=False)
    print(f"[✓] Saved error trend CSV to: {save_path}")

def recurring_test_time_adaptation(cfg):
    # Building model, optimizer and adapter:
    model = build_model(cfg)
    
    # Building optimizer
    optimizer = build_optimizer(cfg)
    
    # Initializing TTA adapter
    tta_adapter = build_adapter(cfg)
    tta_model = tta_adapter(cfg, model, optimizer)
    tta_model.cuda()
    
    # Building data loader
    loader, processor = build_loader(cfg, cfg.CORRUPTION.DATASET, cfg.CORRUPTION.TYPE, cfg.CORRUPTION.SEVERITY)
    
    # Save logs
    outputs_arr = []
    labels_arr = []
    batch_error_rates = []
    
    # Main test-time Adaptation loop 
    tbar = tqdm(loader, dynamic_ncols=True, leave=True, ncols=100)
    for batch_id, data_package in enumerate(tbar):
        data, label, domain = data_package["image"], data_package['label'], data_package['domain']

        if len(label) == 1: 
            continue  # ignore the final single point
        
        data, label = data.cuda(), label.cuda()
        meta = {"label": label, "domain": domain}
        output = tta_model(data, label=meta)
        
        outputs_arr.append(output.detach().cpu().numpy())
        labels_arr.append(label.detach().cpu().numpy())
        
        predict = torch.argmax(output, dim=1)
        accurate = (predict == label)
        error_rate = 1.0 - accurate.float().mean().item()
        batch_error_rates.append(error_rate)
        processor.process(accurate, domain)

        tbar.set_postfix(acc=processor.cumulative_acc())

    labels_arr = np.concatenate(labels_arr, axis=0)
    outputs_arr = np.concatenate(outputs_arr, axis=0)

    processor.calculate()
    _, prcss_eval_csv = processor.info()

    # plotting and saving error trend
    plot_error_trend(error_rates=batch_error_rates, output_dir=cfg.OUTPUT_DIR)
    save_error_trend_to_csv(error_rates=batch_error_rates, output_dir=cfg.OUTPUT_DIR)

    return prcss_eval_csv, tta_model

def main():
    parser = argparse.ArgumentParser("Pytorch Implementation for Test Time Adaptation!")
    parser.add_argument(
        '-acfg',
        '--adapter-config-file',
        metavar="FILE",
        default="",
        help="path to adapter config file",
        type=str)
    parser.add_argument(
        '-dcfg',
        '--dataset-config-file',
        metavar="FILE",
        default="",
        help="path to dataset config file",
        type=str)
    parser.add_argument(
        'opts',
        help='modify the configuration by command line',
        nargs=argparse.REMAINDER,
        default=None)
    
    # Parsing arguments
    args = parser.parse_args()
    if len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')
    cfg.merge_from_file(args.adapter_config_file)
    cfg.merge_from_file(args.dataset_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    setproctitle(f"TTA:{cfg.CORRUPTION.DATASET:>8s}:{cfg.ADAPTER.NAME:<10s}")

    # For reproducibility
    torch.backends.cudnn.benchmark = True
    set_random_seed(cfg.SEED)

    # Running recurring TTA
    prcss_eval_csv, _ = recurring_test_time_adaptation(cfg) 
  
    # Saving evaluation results to files:
    if cfg.OUTPUT_DIR:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok =True)
    
    log_file_name = "%s_%s" % (
        osp.basename(args.dataset_config_file).split('.')[0], 
        osp.basename(args.adapter_config_file).split('.')[0])
    
    with open(osp.join(cfg.OUTPUT_DIR, "%s.csv" % log_file_name), "w") as fo:
        fo.write(prcss_eval_csv)

if __name__ == "__main__":
    main()