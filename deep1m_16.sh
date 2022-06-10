set -ex

python train.py \
    --num_learn 100000 \
    --database deep1m \
    --lambda_uniform 0.05 \
    --dint 1024 --dout 16 \
    --save_best_criterion opq_16,rank=10 \
    --checkpoint_dir test_ckpt/deep1m_16 \
    --validation_quantizers opq_16

python eval.py \
       --database deep1m \
       --quantizer opq_16 \
       --ckpt-path test_ckpt/deep1m_16/checkpoint.pth.best
