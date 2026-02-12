#!/bin/bash
#SBATCH --job-name=cnn_stress_test
#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=cnn_stress_test_%j.log

echo "=========================================="
echo "CNN STRESS-TESTING ASSIGNMENT"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

# ================= USER =================

USERNAME="Your user name"
# ========================================

# -------- SCRATCH DIRECTORIES ----------
export SCRATCH_DIR="/scratch/data/${USERNAME}"
export PROJECT_DIR="${SCRATCH_DIR}/cnn-stress-test"
export DATA_DIR="${PROJECT_DIR}/data"
export CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
export RESULTS_DIR="${PROJECT_DIR}/results"
export FIGURES_DIR="${PROJECT_DIR}/figures"

mkdir -p $DATA_DIR $CHECKPOINT_DIR $RESULTS_DIR $FIGURES_DIR

echo "Project dir: $PROJECT_DIR"
echo "Data dir: $DATA_DIR"
echo "Checkpoints dir: $CHECKPOINT_DIR"
echo "Results dir: $RESULTS_DIR"
echo "Figures dir: $FIGURES_DIR"

# -------- UNBUFFERED OUTPUT -------------
export PYTHONUNBUFFERED=1

# -------- CONDA SETUP (IMPORTANT) -------
export CONDARC=/scratch/data/Your user name/conda/condarc

module purge
module load anaconda3/2024
source ~/.bashrc
conda activate dlops

echo "Loaded modules:"
module list

# -------- INSTALL DEPENDENCIES ----------
echo "Installing/checking dependencies..."
pip install tqdm seaborn scikit-learn opencv-python --quiet

# -------- GPU + TORCH CHECK -------------
echo "=========== GPU INFO ==================="
nvidia-smi
python - <<EOF
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
EOF
echo "========================================"

# -------- GO TO WORK DIR ----------------
cd $PROJECT_DIR || exit 1

echo "Current directory: $(pwd)"
echo "=========================================="

# -------- PHASE 1: BASELINE TRAINING ----
echo ""
echo "=========================================="
echo "PHASE 1: Training Baseline ResNet-18"
echo "=========================================="
python -u train.py
echo "Baseline training completed at: $(date)"

# -------- PHASE 2: BASELINE EVALUATION --
echo ""
echo "=========================================="
echo "PHASE 2: Evaluating Baseline Model"
echo "=========================================="
python -u evaluate.py --checkpoint checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth
echo "Baseline evaluation completed at: $(date)"

# -------- PHASE 3: BASELINE GRAD-CAM ----
echo ""
echo "=========================================="
echo "PHASE 3: Generating Baseline Grad-CAM"
echo "=========================================="
python -u visualize.py --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth
echo "Baseline Grad-CAM completed at: $(date)"

# -------- PHASE 4: CUTOUT TRAINING ------
echo ""
echo "=========================================="
echo "PHASE 4: Training Cutout-Enhanced Model"
echo "=========================================="
python -u train.py --cutout
echo "Cutout training completed at: $(date)"

# -------- PHASE 5: CUTOUT EVALUATION ----
echo ""
echo "=========================================="
echo "PHASE 5: Evaluating Cutout Model"
echo "=========================================="
python -u evaluate.py --checkpoint checkpoints/ResNet-18-CIFAR_cutout_seed42_final.pth --cutout
echo "Cutout evaluation completed at: $(date)"

# -------- PHASE 6: COMPARISON -----------
echo ""
echo "=========================================="
echo "PHASE 6: Comparative Analysis & Visualization"
echo "=========================================="
python -u visualize.py \
    --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth \
    --cutout checkpoints/ResNet-18-CIFAR_cutout_seed42_final.pth
echo "Comparison completed at: $(date)"

# -------- PHASE 7: FINAL COMPARISON -----
echo ""
echo "=========================================="
echo "PHASE 7: Generating Final Summary"
echo "=========================================="
python -u main.py --mode compare \
    --baseline checkpoints/ResNet-18-CIFAR_baseline_seed42_final.pth \
    --cutout checkpoints/ResNet-18-CIFAR_cutout_seed42_final.pth
echo "Final summary completed at: $(date)"


# ===  DISAGREEMENT ANALYSIS ===
echo ""
echo "=========================================="
echo "GENERATING MASSIVE DISAGREEMENT PLOT"
echo "=========================================="
python -u analyze_disagreements.py
# =================================

# -------- LIST OUTPUTS ------------------
echo ""
echo "=========================================="
echo "GENERATED OUTPUTS"
echo "=========================================="
echo "Checkpoints:"
ls -la $CHECKPOINT_DIR
echo ""
echo "Results:"
ls -la $RESULTS_DIR
echo ""
echo "Figures:"
ls -la $FIGURES_DIR
find $FIGURES_DIR -name "*.png" | head -20

# -------- COMPLETION --------------------
echo ""
echo "=========================================="
echo "JOB COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "SUMMARY OF KEY FILES:"
echo "  - Training curves: ${FIGURES_DIR}/*_training_curves.png"
echo "  - Confusion matrices: ${FIGURES_DIR}/*_confusion_matrix.png"
echo "  - Grad-CAM visualizations: ${FIGURES_DIR}/gradcam_*/"
echo "  - Failure analysis: ${RESULTS_DIR}/*_failures/"
echo "  - Comparison summary: ${FIGURES_DIR}/final_comparison_summary.png"
echo "=========================================="
