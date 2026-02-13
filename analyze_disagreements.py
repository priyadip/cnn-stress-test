"""Generates a SINGLE, potentially very tall PNG file showing ALL disagreements."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import config
from models.resnet import resnet18_cifar
from utils.data import get_cifar10_loaders, get_test_dataset_with_indices, denormalize
from evaluate import evaluate_model

def plot_massive_grid(dataset, idx_base_wins, idx_cut_wins, 
                      preds_base, preds_cut, 
                      class_names, save_path):
    """
    Plots a massive vertical grid of all disagreements.
    """
    # Grid settings
    COLS = 10  # 10 images per row
    
    # Calculate rows needed
    rows_base = math.ceil(len(idx_base_wins) / COLS)
    rows_cut = math.ceil(len(idx_cut_wins) / COLS)
    
    # Add buffer rows for titles/spacing
    total_rows = rows_base + rows_cut + 4 
    
    # DYNAMIC HEIGHT: 2.5 inches per row ensures images remain large
    fig_width = 24
    fig_height = total_rows * 2.5 
    
    print(f"Generating massive image: {fig_width}x{fig_height} inches (Rows: {total_rows})")
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # --- HELPER TO PLOT ONE IMAGE ---
    def plot_sub(index_in_grid, data_idx, title_color):
        ax = fig.add_subplot(total_rows, COLS, index_in_grid)
        img, target, _ = dataset[data_idx]
        
        # Denormalize
        img_vis = denormalize(img).permute(1, 2, 0).numpy()
        img_vis = np.clip(img_vis, 0, 1)
        
        ax.imshow(img_vis)
        true_lbl = class_names[target]
        base_lbl = class_names[preds_base['predictions'][data_idx]]
        cut_lbl = class_names[preds_cut['predictions'][data_idx]]
        
        ax.set_title(f"Idx:{data_idx} T:{true_lbl}\nB:{base_lbl} | C:{cut_lbl}", 
                     fontsize=8, color=title_color, fontweight='bold')
        ax.axis('off')

    # --- SECTION 1: BASELINE WINS (Green Title) ---
    # Title Row
    plt.figtext(0.5, 1 - (1/total_rows), 
                f"SCENARIO 1: Baseline Correct / Cutout Fails ({len(idx_base_wins)} images)", 
                ha='center', va='top', fontsize=20, fontweight='bold', color='darkgreen')
    
    current_row = 1 # Start after title margin
    for i, idx in enumerate(idx_base_wins):
        # Calculate grid position
        # row_offset = current_row + (i // COLS)
        # col_offset = (i % COLS) + 1
        plot_index = (current_row * COLS) + i + 1
        plot_sub(plot_index, idx, 'darkgreen')

    # --- SECTION 2: CUTOUT WINS (Blue Title) ---
    # Calculate start row for section 2
    sec2_start_row = current_row + rows_base + 1
    
    # Title for Section 2
    # We place text relative to the figure height
    y_pos = 1 - (sec2_start_row / total_rows)
    plt.figtext(0.5, y_pos, 
                f"SCENARIO 2: Cutout Correct / Baseline Fails ({len(idx_cut_wins)} images)", 
                ha='center', va='bottom', fontsize=20, fontweight='bold', color='navy')

    current_row = sec2_start_row + 1
    for i, idx in enumerate(idx_cut_wins):
        plot_index = (current_row * COLS) + i + 1
        plot_sub(plot_index, idx, 'navy')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust for titles
    plt.savefig(save_path, dpi=100) # dpi=100 keeps file size reasonable for huge dims
    print(f"Saved massive analysis to: {save_path}")
    plt.close()

def main(baseline_ckpt, cutout_ckpt):
    device = config.DEVICE
    print(f"Analyzing disagreements...")

    # Load Data
    _, test_loader = get_cifar10_loaders(use_cutout=False) 
    raw_dataset = get_test_dataset_with_indices() 

    # Load Models
    print("Loading models...")
    baseline = resnet18_cifar(num_classes=config.NUM_CLASSES).to(device)
    baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device, weights_only=True))
    baseline.eval()
    
    cutout = resnet18_cifar(num_classes=config.NUM_CLASSES).to(device)
    cutout.load_state_dict(torch.load(cutout_ckpt, map_location=device, weights_only=True))
    cutout.eval()

    # Get Predictions
    print("Evaluating Baseline...")
    base_res = evaluate_model(baseline, test_loader, device)
    print("Evaluating Cutout...")
    cut_res = evaluate_model(cutout, test_loader, device)

    # Find Intersections
    base_correct = np.array(base_res['correct'])
    cut_correct = np.array(cut_res['correct'])
    
    # 1. Baseline Wins
    base_wins_mask = (base_correct == True) & (cut_correct == False)
    idx_base_wins = np.where(base_wins_mask)[0]
    
    # 2. Cutout Wins
    cut_wins_mask = (cut_correct == True) & (base_correct == False)
    idx_cut_wins = np.where(cut_wins_mask)[0]

    print(f"Found {len(idx_base_wins)} cases where Baseline wins.")
    print(f"Found {len(idx_cut_wins)} cases where Cutout wins.")

    # Visualize Combined
    save_dir = os.path.join(config.FIGURES_DIR, 'disagreements')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'massive_disagreement_sheet.png')

    plot_massive_grid(
        raw_dataset, idx_base_wins, idx_cut_wins,
        base_res, cut_res,
        config.CLASS_NAMES, save_path
    )

if __name__ == "__main__":
    base_path = os.path.join(config.CHECKPOINT_DIR, f"{config.get_experiment_name(False)}_final.pth")
    cut_path = os.path.join(config.CHECKPOINT_DIR, f"{config.get_experiment_name(True)}_final.pth")
    main(base_path, cut_path)
