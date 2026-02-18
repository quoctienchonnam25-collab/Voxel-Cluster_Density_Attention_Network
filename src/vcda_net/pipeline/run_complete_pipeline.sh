#!/usr/bin/env bash
"""
Complete Pipeline for Regional Brain Age Prediction

This pipeline consists of 3 steps:
1. Extract regional features from trained model
2. Train regional predictors with bias correction
3. Generate predictions on new datasets

Author: Anonymous
Date: 2026-02-17
"""

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Model checkpoints
CHECKPOINT="${PACKAGE_ROOT}/vcda_runs/vcda_net_20260216_011913/checkpoints/best_model.pth"
UNET_CHECKPOINT="${PACKAGE_ROOT}/unet_checkpoint/IXI_3dunet_best_model.pth"

# Data paths (MODIFY THESE FOR YOUR DATA)
DATA_DIR="/media/devin/WORK/devin/tien/synthesis_data/healthy_brain_1710/ABIDE_ADNI_IXI_OASIS_PPMI_Turboprep_balanced_1710"
METADATA_CSV="/media/devin/WORK/devin/tien/synthesis_data/healthy_brain_1710/ABIDE_ADNI_IXI_OASIS_PPMI_Turboprep_balanced_1710_metadata.csv"

# Output directories
OUTPUT_BASE="${PACKAGE_ROOT}/pipeline_output"
FEATURES_DIR="${OUTPUT_BASE}/regional_features"
MODELS_DIR="${OUTPUT_BASE}/regional_predictors"

# Parameters
ALPHA=1.0  # Ridge regression parameter
RANDOM_SEED=42
DEVICE="cuda"

# ==============================================================================
# SETUP
# ==============================================================================

echo "================================================================================"
echo "VCDA-NET - COMPLETE PIPELINE"
echo "================================================================================"
echo ""
echo "Pipeline Steps:"
echo "  1. Extract regional features from trained model"
echo "  2. Train regional predictors with bias correction"
echo "  3. Ready for prediction on new datasets"
echo ""
echo "Configuration:"
echo "  Checkpoint: $(basename $CHECKPOINT)"
echo "  Data: $(basename $DATA_DIR)"
echo "  Output: $OUTPUT_BASE"
echo "  Device: $DEVICE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$FEATURES_DIR"
mkdir -p "$MODELS_DIR"

# Activate conda environment
echo "Activating conda environment..."
source ~/devin/programs/anaconda3/bin/activate base

# Add package to PYTHONPATH
export PYTHONPATH="${PACKAGE_ROOT}/src:${PYTHONPATH}"

# ==============================================================================
# STEP 1: EXTRACT REGIONAL FEATURES
# ==============================================================================

echo ""
echo "================================================================================"
echo "STEP 1/2: EXTRACT REGIONAL FEATURES"
echo "================================================================================"
echo ""

python -m vcda_net.pipeline.extract_regional_features \
    --checkpoint "$CHECKPOINT" \
    --unet_checkpoint "$UNET_CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --metadata "$METADATA_CSV" \
    --output_dir "$FEATURES_DIR" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "✗ Feature extraction failed!"
    exit 1
fi

echo ""
echo "✓ Feature extraction complete!"

# ==============================================================================
# STEP 2: TRAIN REGIONAL PREDICTORS
# ==============================================================================

echo ""
echo "================================================================================"
echo "STEP 2/2: TRAIN REGIONAL PREDICTORS"
echo "================================================================================"
echo ""

python -m vcda_net.pipeline.train_regional_predictors \
    --features_dir "$FEATURES_DIR" \
    --output_dir "$MODELS_DIR" \
    --alpha "$ALPHA" \
    --random_seed "$RANDOM_SEED"

if [ $? -ne 0 ]; then
    echo "✗ Training failed!"
    exit 1
fi

echo ""
echo "✓ Training complete!"

# ==============================================================================
# COMPLETION
# ==============================================================================

echo ""
echo "================================================================================"
echo "                                                                                "
echo "                         PIPELINE COMPLETED SUCCESSFULLY!                       "
echo "                                                                                "
echo "================================================================================"
echo ""
echo "Output structure:"
echo "  $OUTPUT_BASE/"
echo "  ├── regional_features/"
echo "  │   ├── regional_features.npy"
echo "  │   ├── metadata.csv"
echo "  │   ├── extraction_summary.json"
echo "  │   └── regions/ (32 .npy files)"
echo "  │"
echo "  └── regional_predictors/"
echo "      ├── models/ (32 .pkl files)"
echo "      ├── bias_correction/"
echo "      │   └── regional_bias_correction_params.json"
echo "      ├── data_split.json"
echo "      ├── regional_predictors_results.csv"
echo "      └── training_summary.json"
echo ""
echo "Next steps:"
echo "  1. Review training results in $MODELS_DIR/regional_predictors_results.csv"
echo "  2. Use trained models for prediction on new datasets"
echo "  3. See experiments/ad_prediction/ for example usage"
echo ""
echo "================================================================================"
