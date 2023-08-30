# !/bin/bash

python iemocap_gan_ffn.py --seed 4207 --no-cuda --b1 0.5 --b2 0.999 \
    --lr 0.0001 --l2 0.008 --dropout 0.6 --batch-size 32 --epochs 160 --GAN-epochs 30

