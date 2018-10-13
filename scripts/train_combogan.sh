python train.py  \
    --dataroot ./datasets/robotcar  \
    --name robotcar_night2day  \
    --n_domains 2  \
    --niter 25  --niter_decay 25  \
    --loadSize 512  --fineSize 384
