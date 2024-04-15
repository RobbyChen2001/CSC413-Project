#!/bin/bash

source venv/bin/activate

cd ZSE-SBIR || exit

python train.py --data_path ./datasets --dataset sketchy_extend --test_class test_class_sketchy25 --epoch 10 -s ./checkpoints/sketchy_ext -c 0 -r rn
python test.py --data_path ./datasets --dataset sketchy_extend --test_class test_class_sketchy25 -l ./checkpoints/sketchy_ext/best_checkpoint.pth -c 0 -r rn
python test.py --data_path ./datasets --dataset sketchy_extend --test_class test_class_sketchy25 -l ./checkpoints/sketchy_ext/best_checkpoint.pth -c 0 -r rn --test_augmentation gaussian-noise
python test.py --data_path ./datasets --dataset sketchy_extend --test_class test_class_sketchy25 -l ./checkpoints/sketchy_ext/best_checkpoint.pth -c 0 -r rn --test_augmentation rotation
python test.py --data_path ./datasets --dataset sketchy_extend --test_class test_class_sketchy25 -l ./checkpoints/sketchy_ext/best_checkpoint.pth -c 0 -r rn --test_augmentation anisotropic-diffusion
python test.py --data_path ./datasets --dataset sketchy_extend --test_class test_class_sketchy25 -l ./checkpoints/sketchy_ext/best_checkpoint.pth -c 0 -r rn --test_augmentation translation
python test.py --data_path ./datasets --dataset sketchy_extend --test_class test_class_sketchy25 -l ./checkpoints/sketchy_ext/best_checkpoint.pth -c 0 -r rn --test_augmentation sharpen

cd .. || exit

deactivate
