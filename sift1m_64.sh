set -ex

python train.py \
    --num_learn 100000 \
    --database sift1m \
    --lambda_uniform 0.02 \
    --dint 1024 --dout 24 \
    --save_best_criterion opq_64,rank=10 \
    --checkpoint_dir test_ckpt/sift1m_64 \
    --validation_quantizers opq_64

python eval.py \
       --database sift1m \
       --quantizer opq_64 \
       --ckpt-path test_ckpt/sift1m_64/checkpoint.pth.best
