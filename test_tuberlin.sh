#!/bin/bash

source venv/bin/activate

cd ZSE-SBIR || exit

python train.py --data_path ./datasets --dataset tu_berlin --test_class test_class_tuberlin30 --epoch 10 -s ./checkpoints/tuberlin_ext
python test.py --data_path ./datasets --dataset tu_berlin --test_class test_class_tuberlin30 -l ./checkpoints/tuberlin_ext/best_checkpoint.pth
python test.py --data_path ./datasets --dataset tu_berlin --test_class test_class_tuberlin30 -l ./checkpoints/tuberlin_ext/best_checkpoint.pth --test_augmentation gaussian-noise
python test.py --data_path ./datasets --dataset tu_berlin --test_class test_class_tuberlin30 -l ./checkpoints/tuberlin_ext/best_checkpoint.pth --test_augmentation rotation
python test.py --data_path ./datasets --dataset tu_berlin --test_class test_class_tuberlin30 -l ./checkpoints/tuberlin_ext/best_checkpoint.pth --test_augmentation anisotropic-diffusion
python test.py --data_path ./datasets --dataset tu_berlin --test_class test_class_tuberlin30 -l ./checkpoints/tuberlin_ext/best_checkpoint.pth --test_augmentation sharpen

cd .. || exit

deactivate
