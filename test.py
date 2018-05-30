import time
import os
from options.test_options import TestOptions
from data.data_loader import DataLoader
from models.combogan_model import ComboGANModel
from util.visualizer import Visualizer
from util import html


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1

dataset = DataLoader(opt)
model = ComboGANModel(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%d' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %d' % (opt.name, opt.phase, opt.which_epoch))
# store images for matrix visualization
vis_buffer = []

# test
for i, data in enumerate(dataset):
    if not opt.serial_test and i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals(testing=True)
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

    if opt.show_matrix:
        vis_buffer.append(visuals)
        if (i+1) % opt.n_domains == 0:
            save_path = os.path.join(web_dir, 'mat_%d.png' % (i//opt.n_domains))
            visualizer.save_image_matrix(vis_buffer, save_path)
            vis_buffer.clear()

webpage.save()

