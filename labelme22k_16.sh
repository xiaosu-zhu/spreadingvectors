set -ex

python train.py \
    --num_learn 100000 \
    --database labelme22k \
    --lambda_uniform 0.05 \
    --dint 1024 --dout 16 \
    --save_best_criterion opq_16,rank=10 \
    --checkpoint_dir test_ckpt/labelme22k_16 \
    --validation_quantizers opq_16

python eval.py \
       --database labelme22k \
       --quantizer opq_16 \
       --ckpt-path test_ckpt/labelme22k_16/checkpoint.pth.best
