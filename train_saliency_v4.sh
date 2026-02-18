#!/bin/bash

###############################################################################
# Training Script for Hybrid Saliency Enhanced V4
# 
# This script trains the HybridSaliencyEnhanced model with saliency map
# enhanced features and gated fusion architecture.
#
# Usage:
#   bash train_saliency_v4.sh
#
# Author: Anonymous
# Date: 2026-02-15
# Version: 4.0.2
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}                                                                                ${NC}"
echo -e "${BLUE}         Training Hybrid Saliency Enhanced V4 - Brain Age Prediction           ${NC}"
echo -e "${BLUE}                                                                                ${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="/media/devin/WORK/devin/tien/synthesis_data/healthy_brain_1710/ABIDE_ADNI_IXI_OASIS_PPMI_Turboprep_balanced_1710"
METADATA_CSV="/media/devin/WORK/devin/tien/synthesis_data/healthy_brain_1710/ABIDE_ADNI_IXI_OASIS_PPMI_Turboprep_balanced_1710_metadata.csv"
UNET_CHECKPOINT="${SCRIPT_DIR}/src/hybrid_saliency_v4/checkpoints/IXI_3dunet_best_model.pth"

# Training parameters
EPOCHS=300
BATCH_SIZE=4
LEARNING_RATE=1e-3
WEIGHT_DECAY=1e-4
DROPOUT=0.3

# Model parameters
NUM_REGIONS=32
EMBEDDING_DIM=256
RESNET_DEPTH="resnet18"

# Saliency Map parameters
TOP_K=128
SIGMA=10.0
MATRIX_RESIZE=64

# GNN parameters
EDGE_NUM=31
HIDDEN_CHANNELS=64
NUM_GNN_LAYERS=3
USE_EDGE_ATTENTION="--use_edge_attention"

# Transformer parameters
TRANSFORMER_D_MODEL=256
TRANSFORMER_NHEAD=8
TRANSFORMER_NUM_LAYERS=3

# Loss function
LOSS_TYPE="huber"
HUBER_DELTA=1.0

# Data split
TRAIN_RATIO=0.7
VAL_RATIO=0.15
STRATIFY="--stratify"
RANDOM_SEED=42

# UNet fine-tuning
FREEZE_UNET="--unfreeze_unet"  # "--freeze_unet" or "--unfreeze_unet" to enable fine-tuning
UNET_LR="--unet_lr 1e-5"  # Set to "--unet_lr 1e-5" if unfreezing UNet
UNFREEZE_AFTER_EPOCH=0  # Set to >0 to gradually unfreeze (e.g., 50)

# Early stopping
EARLY_STOP="--early_stop"
PATIENCE=25

# Output
OUTPUT_DIR="saliency_runs"
RUN_NAME=""  # Leave empty for auto-generated timestamp name

# Device
DEVICE="cuda"

# ============================================================================
# VALIDATION
# ============================================================================

echo -e "${YELLOW}Validating paths...${NC}"

if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}❌ Error: Data directory not found: $DATA_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data directory found${NC}"

if [ ! -f "$METADATA_CSV" ]; then
    echo -e "${RED}❌ Error: Metadata CSV not found: $METADATA_CSV${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Metadata CSV found${NC}"

if [ ! -f "$UNET_CHECKPOINT" ]; then
    echo -e "${RED}❌ Error: UNet checkpoint not found: $UNET_CHECKPOINT${NC}"
    exit 1
fi
echo -e "${GREEN}✓ UNet checkpoint found${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ python3 found: $(python3 --version)${NC}"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA available${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠ Warning: nvidia-smi not found, will use CPU${NC}"
    DEVICE="cpu"
fi

echo ""

# ============================================================================
# TRAINING COMMAND
# ============================================================================

echo -e "${BLUE}Starting training with the following configuration:${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Data directory:    $DATA_DIR"
echo "  Metadata CSV:      $METADATA_CSV"
echo "  UNet checkpoint:   $UNET_CHECKPOINT"
echo ""
echo "  Epochs:            $EPOCHS"
echo "  Batch size:        $BATCH_SIZE"
echo "  Learning rate:     $LEARNING_RATE"
echo "  Loss type:         $LOSS_TYPE"
echo "  Device:            $DEVICE"
echo ""
echo "  UNet frozen:       ${FREEZE_UNET:2}"  # Remove "--" prefix
echo "  Early stopping:    ${EARLY_STOP:2} (patience=$PATIENCE)"
echo "  Output directory:  $OUTPUT_DIR"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Build run name argument
RUN_NAME_ARG=""
if [ -n "$RUN_NAME" ]; then
    RUN_NAME_ARG="--run_name $RUN_NAME"
fi

# Build UNet LR argument
UNET_LR_ARG=""
if [ -n "$UNET_LR" ]; then
    UNET_LR_ARG="$UNET_LR"
fi

# Confirm before starting
# read -p "Press Enter to start training (or Ctrl+C to cancel)..."
echo "Starting training now..."
echo ""

# ============================================================================
# RUN TRAINING
# ============================================================================

cd "$SCRIPT_DIR"

# Activate conda base environment (where PyTorch is installed)
echo "Activating conda environment..."
source ~/devin/programs/anaconda3/bin/activate base
# conda activate base

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Add src to PYTHONPATH so Python can find the module
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

python -m hybrid_saliency_v4.training.train \
    --data_dir "$DATA_DIR" \
    --metadata_csv "$METADATA_CSV" \
    --unet_checkpoint "$UNET_CHECKPOINT" \
    \
    --num_regions $NUM_REGIONS \
    --embedding_dim $EMBEDDING_DIM \
    --resnet_depth $RESNET_DEPTH \
    \
    $FREEZE_UNET \
    $UNET_LR_ARG \
    --unfreeze_unet_after_epoch $UNFREEZE_AFTER_EPOCH \
    \
    --top_k $TOP_K \
    --sigma $SIGMA \
    --matrix_resize $MATRIX_RESIZE \
    \
    --edge_num $EDGE_NUM \
    --hidden_channels $HIDDEN_CHANNELS \
    --num_gnn_layers $NUM_GNN_LAYERS \
    $USE_EDGE_ATTENTION \
    \
    --transformer_d_model $TRANSFORMER_D_MODEL \
    --transformer_nhead $TRANSFORMER_NHEAD \
    --transformer_num_layers $TRANSFORMER_NUM_LAYERS \
    \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --dropout $DROPOUT \
    \
    --loss_type $LOSS_TYPE \
    --huber_delta $HUBER_DELTA \
    \
    --train_ratio $TRAIN_RATIO \
    --val_ratio $VAL_RATIO \
    $STRATIFY \
    --random_seed $RANDOM_SEED \
    \
    $EARLY_STOP \
    --patience $PATIENCE \
    \
    --output_dir "$OUTPUT_DIR" \
    $RUN_NAME_ARG \
    --device $DEVICE

# ============================================================================
# COMPLETION
# ============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}                                                                                ${NC}"
    echo -e "${GREEN}                   TRAINING COMPLETED SUCCESSFULLY!                             ${NC}"
    echo -e "${GREEN}                                                                                ${NC}"
    echo -e "${GREEN}================================================================================${NC}"
    echo ""
    echo -e "${BLUE}Results saved to: $OUTPUT_DIR/${NC}"
    echo ""
    echo -e "${BLUE}To view training progress:${NC}"
    echo "  tensorboard --logdir=$OUTPUT_DIR"
    echo ""
else
    echo ""
    echo -e "${RED}================================================================================${NC}"
    echo -e "${RED}                                                                                ${NC}"
    echo -e "${RED}                          TRAINING FAILED!                                      ${NC}"
    echo -e "${RED}                                                                                ${NC}"
    echo -e "${RED}================================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Check the error messages above for details.${NC}"
    exit 1
fi
