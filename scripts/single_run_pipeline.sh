#!/bin/bash

# Exit on any error
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Set working directory
cd "$PROJECT_ROOT"
# Add project to Python path
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Parse command line arguments
CITY="Toronto"
CONFIG="config.yaml"
SKIP_TRAIN=false
MODEL_CITY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --city)
            CITY="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --model-city)
            MODEL_CITY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo " Starting Pipeline for City: $CITY"
echo "================================================================================"

# Build arguments
TRAIN_ARGS="--city $CITY --config $CONFIG"
EVAL_ARGS="--city $CITY --config $CONFIG"

if [ ! -z "$MODEL_CITY" ]; then
    EVAL_ARGS="$EVAL_ARGS --model-city $MODEL_CITY"
fi

# --- STEP: TRAINING ---
if [ "$SKIP_TRAIN" = false ] && [ -z "$MODEL_CITY" ]; then
    echo ""
    echo "================================================================================"
    echo " STEP: Training Model"
    echo "================================================================================"
    python scripts/train_single_run.py $TRAIN_ARGS
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed!"
        exit 1
    fi
    echo "✓ Training Model finished"
else
    echo ""
    echo "ℹ️  Skipping Training"
fi

# --- STEP: CONVEX HULL IMAGES (Only when not training or using external model) ---
if [ "$SKIP_TRAIN" = true ] || [ ! -z "$MODEL_CITY" ]; then
    # Only eval set when using pre-trained or external model
    echo ""
    echo "================================================================================"
    echo " STEP: Generating Convex Hull Images (Evaluation Set Only)"
    echo "================================================================================"
    python scripts/generate_convexhull.py $EVAL_ARGS --set eval
    if [ $? -ne 0 ]; then
        echo "ERROR: Convex hull generation failed!"
        exit 1
    fi
    echo "✓ Generating Convex Hull Images finished"
else
    # Both train and eval sets after training
    echo ""
    echo "================================================================================"
    echo " STEP: Generating Convex Hull Images (Training and Evaluation Sets)"
    echo "================================================================================"
    python scripts/generate_convexhull.py $TRAIN_ARGS --set all
    if [ $? -ne 0 ]; then
        echo "ERROR: Convex hull generation failed!"
        exit 1
    fi
    echo "✓ Generating Convex Hull Images finished"
fi
# --- STEP: VISUALIZATION (Post-Processing Images) ---
echo ""
echo "================================================================================"
echo " STEP: Generating Post-Processing Images"
echo "================================================================================"
python scripts/generate_final_images.py $EVAL_ARGS --set eval
if [ $? -ne 0 ]; then
    echo "ERROR: Post-processing image generation failed!"
    exit 1
fi
echo " Generating Post-Processing Images finished"



# --- STEP: METRICS ---
echo ""
echo "================================================================================"
echo " STEP: Calculating Final Metrics"
echo "================================================================================"
python scripts/one_shot.py $EVAL_ARGS
if [ $? -ne 0 ]; then
    echo "ERROR: Metrics calculation failed!"
    exit 1
fi
echo " Calculating Final Metrics finished"

echo ""
echo "================================================================================"
echo " PIPELINE COMPLETED SUCCESSFULLY!"
echo " Results located in: outputs/$CITY/"
echo "================================================================================"