set -ex

python train.py \
    --num_learn 100000 \
    --database labelme22k \
    --lambda_uniform 0.02 \
    --dint 1024 --dout 24 \
    --save_best_criterion opq_64,rank=10 \
    --checkpoint_dir test_ckpt/labelme22k_64 \
    --validation_quantizers opq_64

python eval.py \
       --database labelme22k \
       --quantizer opq_64 \
       --ckpt-path test_ckpt/labelme22k_64/checkpoint.pth.best
