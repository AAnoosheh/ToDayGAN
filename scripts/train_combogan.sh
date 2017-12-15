python train.py  \
    --dataroot ./datasets/alps  \
    --name alps_combogan  \
    --n_domains 4  \
    --niter 200  \
    --niter_decay 200  \
    --lambda_identity 0.0  \
    --lambda_forward 0.0
