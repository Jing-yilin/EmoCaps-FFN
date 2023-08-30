clear
python iemocap_lstm.py --seed 4207 --no-cuda --lr 0.0001 --l2 0.00001\
 --dropout 0.25 --batch-size 32 --epochs 80
 # --class-weight --attention --tensorboard

# tensorboard --logdir runs