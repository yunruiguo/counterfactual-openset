#!/bin/bash
# Break on any error
set -e

DATASET_DIR=../data

# Download any datasets not currently available
# TODO: do this in python, based on --dataset

if [ ! -f $DATASET_DIR/svhn-split0a.dataset ]; then
    python generativeopenset/datasets/download_svhn.py
fi

# Hyperparameters
GAN_EPOCHS=0
CLASSIFIER_EPOCHS=3
CF_COUNT=50
GENERATOR_MODE=counterfactual


# Train the intial generative model (E+G+D) and the initial classifier (C_K)
python ./generativeopenset/train_gan.py --epochs $GAN_EPOCHS

# Baseline: Evaluate the standard classifier (C_k+1)
python ./generativeopenset/evaluate_classifier.py --result_dir . --mode baseline
python ./generativeopenset/evaluate_classifier.py --result_dir . --mode weibull

GAN_EPOCHS=132

cp ./checkpoints/classifier_k_epoch_0${GAN_EPOCHS}.pth ./checkpoints/classifier_kplusone_epoch_0${GAN_EPOCHS}.pth

# Generate a number of counterfactual images (in the K+2 by K+2 square grid format)
python ./generativeopenset/generate_${GENERATOR_MODE}.py --result_dir . --count $CF_COUNT

# Automatically label the rightmost column in each grid (ignore the others)
python ./generativeopenset/auto_label.py --output_filename generated_images_${GENERATOR_MODE}.dataset

# Train a new classifier, now using the aux_dataset containing the counterfactuals
python ./generativeopenset/train_classifier.py --epochs $CLASSIFIER_EPOCHS --aux_dataset generated_images_${GENERATOR_MODE}.dataset

# Evaluate the C_K+1 classifier, trained with the augmented data
python ./generativeopenset/evaluate_classifier.py --result_dir . --mode fuxin

./print_results.sh
