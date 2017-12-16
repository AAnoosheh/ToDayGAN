import time
from options.train_options import TrainOptions
from data.data_loader import DataLoader
from models.combogan_model import ComboGANModel
from util.visualizer import Visualizer


opt = TrainOptions().parse()

dataset = DataLoader(opt)
print('# training images = %d' % len(dataset))
model = ComboGANModel(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.which_epoch + 1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        decay_frac = (epoch - opt.niter) / opt.niter_decay
        new_lr = opt.lr * (1 - decay_frac)
        model.update_learning_rate(new_lr)
