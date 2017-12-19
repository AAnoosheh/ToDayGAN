
# ComboGAN

This is our ongoing PyTorch implementation for ComboGAN.
Code was written by [Asha Anoosheh](https://github.com/aanoosheh) (built upon [CycleGAN](https://github.com/junyanz/CycleGAN))


#### [[ComboGAN Paper]](https://arxiv.org/pdf/1712.06909.pdf)
<img src="img/Inference.png" width=420/>


If you use this code for your research, please cite:

ComboGAN: Unrestrained Scalability for Image Domain Translation
[Asha Anoosheh](http://ashaanoosheh.com),  [Eirikur Augustsson](https://relational.github.io/), [Radu Timofte](http://www.vision.ee.ethz.ch/~timofter/), [Luc van Gool](https://www.vision.ee.ethz.ch/en/members/get_member.cgi?id=1)
In Arxiv, 2017.


<br><br>
<img src='img/Paintings.png' align="center" width=900>
<br><br>


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/AAnoosheh/ComboGAN.git
cd ComboGAN
```

### ComboGAN training
Example scripts can be found in the `scripts` directory.

- Train a model:
```
python train.py --name <experiment_name> --dataroot ./datasets/<your_dataset> --n_domains <N> --niter <num_epochs_constant_LR> --niter_decay <num_epochs_decaying_LR>
```
Checkpoints will be saved by default to `./checkpoints/<experiment_name>/`
- Fine-tuning/Resume training:
```
python train.py --continue_train --which_epoch <checkpoint_number_to_load> --name <experiment_name> --dataroot ./datasets/<your_dataset> --n_domains <N> --niter <num_epochs_constant_LR> --niter_decay <num_epochs_decaying_LR>
```
- Test the model:
```
python test.py --phase test --name <experiment_name> --dataroot ./datasets/<your_dataset> --n_domains <N> --which_epoch <checkpoint_number_to_load>
```
The test results will be saved to a html file here: `./results/<experiment_name>/<epoch_number>/index.html`.



## Training/Testing Details
- Flags: see `options/train_options.py` for training-specific flags; see `options/test_options.py` for test-specific flags; and see `options/base_options.py` for all common flags.
- CPU/GPU (default `--gpu_ids 0`): set`--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g. `--batchSize 32`) to benefit from multiple GPUs.
- Visualization: during training, the current results and loss plots can be viewed using two methods. First, if you set `--display_id` > 0, the results and loss plot will appear on a local graphics web server launched by [visdom](https://github.com/facebookresearch/visdom). To do this, you should have `visdom` installed and a server running by the command `python -m visdom.server`. The default server URL is `http://localhost:8097`. `display_id` corresponds to the window ID that is displayed on the `visdom` server. The `visdom` display functionality is turned on by default. To avoid the extra overhead of communicating with `visdom` set `--display_id 0`. Secondly, the intermediate results are also saved to `./checkpoints/<experiment_name>/web/index.html`. To avoid this, set the `--no_html` flag.
- Preprocessing: images can be resized and cropped in different ways using `--resize_or_crop` option. The default option `'resize_and_crop'` resizes the image to be of size `(opt.loadSize, opt.loadSize)` and does a random crop of size `(opt.fineSize, opt.fineSize)`. `'crop'` skips the resizing step and only performs random cropping. `'scale_width'` resizes the image to have width `opt.fineSize` while keeping the aspect ratio. `'scale_width_and_crop'` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`.


NOTE: one should **not** expect ComboGAN to work on just any combination of input and output datasets (e.g. `dogs<->houses`). We find it works better if two datasets share similar visual content. For example, `landscape painting<->landscape photographs` works much better than `portrait painting <-> landscape photographs`.
