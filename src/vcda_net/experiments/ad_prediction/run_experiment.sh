#!/bin/bash

# ==============================================================================
# Run AD Brain Age Gap Prediction Experiment
# ==============================================================================
# This script runs the regional brain age prediction pipeline on ADNI AD/MCI
# cohort using the trained VCDA-Net model.
# ==============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Data paths (now inside package)
DATA_DIR="${PACKAGE_ROOT}/data/ad_prediction"

FEATURES_DIR="${DATA_DIR}/features/regions"
METADATA_FILE="${DATA_DIR}/features/metadata.csv"
MODELS_DIR="${DATA_DIR}/models/models"
BIAS_PARAMS="${DATA_DIR}/models/bias_correction/regional_bias_correction_params.json"

# Output configuration
OUTPUT_DIR="${SCRIPT_DIR}/results"
OUTPUT_FILE="${OUTPUT_DIR}/predictions_adni_ad_mci.csv"

# Model parameters
NUM_REGIONS=32

# ==============================================================================
# VALIDATION
# ==============================================================================

echo "================================================================================"
echo "AD BRAIN AGE GAP PREDICTION EXPERIMENT"
echo "VCDA-Net Model Application"
echo "================================================================================"
echo ""

echo "Validating paths..."

if [ ! -d "$FEATURES_DIR" ]; then
    echo "✗ Features directory not found: $FEATURES_DIR"
    exit 1
fi
echo "✓ Features directory found"

if [ ! -f "$METADATA_FILE" ]; then
    echo "✗ Metadata file not found: $METADATA_FILE"
    exit 1
fi
echo "✓ Metadata file found"

if [ ! -d "$MODELS_DIR" ]; then
    echo "✗ Models directory not found: $MODELS_DIR"
    exit 1
fi
echo "✓ Models directory found"

if [ ! -f "$BIAS_PARAMS" ]; then
    echo "✗ Bias parameters file not found: $BIAS_PARAMS"
    exit 1
fi
echo "✓ Bias parameters file found"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory ready: $OUTPUT_DIR"

# ==============================================================================
# EXPERIMENT CONFIGURATION SUMMARY
# ==============================================================================

echo ""
echo "Experiment Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Dataset:         ADNI AD/MCI"
echo "  Features:        $(basename $FEATURES_DIR)"
echo "  Models:          $(basename $MODELS_DIR)"
echo "  Regions:         $NUM_REGIONS"
echo "  Bias correction: Enabled (from healthy controls)"
echo "  Output:          $OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ==============================================================================
# ACTIVATE ENVIRONMENT
# ==============================================================================

echo "Activating conda environment..."
# source ~/anaconda3/bin/activate base

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ==============================================================================
# RUN EXPERIMENT
# ==============================================================================

cd "$SCRIPT_DIR"

# Add package to PYTHONPATH
export PYTHONPATH="${PACKAGE_ROOT}/src:${PYTHONPATH}"

echo "Starting experiment..."
echo ""

python -m vcda_net.experiments.ad_prediction.predict_regional_brain_age \
    --features_dir "$FEATURES_DIR" \
    --metadata_file "$METADATA_FILE" \
    --models_dir "$MODELS_DIR" \
    --bias_params_file "$BIAS_PARAMS" \
    --output_file "$OUTPUT_FILE" \
    --num_regions $NUM_REGIONS

# ==============================================================================
# COMPLETION
# ==============================================================================

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "                                                                                "
    echo "                   EXPERIMENT COMPLETED SUCCESSFULLY!                           "
    echo "                                                                                "
    echo "================================================================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    echo "  - predictions_adni_ad_mci.csv"
    echo "  - predictions_adni_ad_mci_statistics.json"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "                                                                                "
    echo "                          EXPERIMENT FAILED!                                    "
    echo "                                                                                "
    echo "================================================================================"
    echo ""
    echo "Check the error messages above for details."
    exit 1
fi
