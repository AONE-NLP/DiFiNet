python train.py -n 80 --lr 1e-6 --num 20 --cnn_dim 120 --msize 5 --biaffine_size 120 --n_head 4 -b 32 -d ace2004 --logit_drop 0.2 --cnn_depth 1 --warmup 0.1 --seed 43
# python train.py -n 60 --lr 1e-6 --num 25 --cnn_dim 120 --msize 5 --biaffine_size 200 --n_head 1 -b 48 -d ace2004 --logit_drop 0.1 --cnn_depth 2 --warmup 0.1 --seed 43 #roberta
