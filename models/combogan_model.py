import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class ComboGANModel(BaseModel):
    def name(self):
        return 'ComboGANModel'

    def __init__(self, opt):
        super(ComboGANModel, self).__init__(opt)

        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.n_domains, opt.norm, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            self.L1 = torch.nn.SmoothL1Loss()
            self.downsample = torch.nn.AvgPool2d(3, stride=2)
            self.criterionCycle = self.L1
            self.criterionIdt = lambda y,t : self.L1(self.downsample(y), self.downsample(t))
            self.criterionLatent = lambda y,t : self.L1(y, Variable(t.data, requires_grad=False))
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            # initialize loss storage
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            self.loss_cycle = [0]*self.n_domains
            # initialize loss multipliers
            self.lambda_cyc, self.lambda_enc = opt.lambda_cycle, (0 * opt.lambda_latent)
            self.lambda_idt, self.lambda_fwd = opt.lambda_identity, opt.lambda_forward

        print('---------- Networks initialized -------------')
        print(self.netG)
        if self.isTrain:
            print(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.DA = input['DA'][0]
        if self.isTrain:
            input_B = input['B']
            self.input_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]
        self.image_paths = input['path']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        real = Variable(self.input_A, volatile=True)
        self.visuals = [real]
        self.labels = ['real_%d' % self.DA]

        # cache encoding to not repeat it everytime
        encoded = self.netG.encode(real, self.DA)
        for d in range(self.n_domains):
            if d is self.DA and not self.opt.autoencode:
                continue
            fake = self.netG.decode(encoded, d)
            self.visuals.append( fake )
            self.labels.append( 'fake_%d' % d )
            if self.opt.reconstruct:
                rec = self.netG.forward(fake, d, self.DA)
                self.visuals.append( rec )
                self.labels.append( 'rec_%d' % d )

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, real, fake, domain):
        # Real
        pred_real = self.netD.forward(real, domain)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        #D_A
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.real_B, fake_B, self.DB)
        #D_B
        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.real_A, fake_A, self.DA)

    def backward_G(self):
        encoded_A = self.netG.encode(self.real_A, self.DA)
        encoded_B = self.netG.encode(self.real_B, self.DB)

        # Optional identity "autoencode" loss
        if self.lambda_idt > 0:
            # Same encoder and decoder should recreate image
            idt_A = self.netG.decode(encoded_A, self.DA)
            loss_idt_A = self.criterionIdt(idt_A, self.real_A)
            idt_B = self.netG.decode(encoded_B, self.DB)
            loss_idt_B = self.criterionIdt(idt_B, self.real_B)
        else:
            loss_idt_A, loss_idt_B = 0, 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG.decode(encoded_A, self.DB)
        pred_fake = self.netD.forward(self.fake_B, self.DB)
        self.loss_G[self.DA] = self.criterionGAN(pred_fake, True)
        # D_B(G_B(B))
        self.fake_A = self.netG.decode(encoded_B, self.DA)
        pred_fake = self.netD.forward(self.fake_A, self.DA)
        self.loss_G[self.DB] = self.criterionGAN(pred_fake, True)
        # Forward cycle loss
        rec_encoded_A = self.netG.encode(self.fake_B, self.DB)
        self.rec_A = self.netG.decode(rec_encoded_A, self.DA)
        self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A)
        # Backward cycle loss
        rec_encoded_B = self.netG.encode(self.fake_A, self.DA)
        self.rec_B = self.netG.decode(rec_encoded_B, self.DB)
        self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B)

        # Optional cycle loss on encoding space
        if self.lambda_enc > 0:
            loss_enc_A = self.criterionLatent(rec_encoded_A, encoded_A)
            loss_enc_B = self.criterionLatent(rec_encoded_B, encoded_B)
        else:
            loss_enc_A, loss_enc_B = 0, 0

        # Optional loss on downsampled image before and after
        if self.lambda_fwd > 0:
            loss_fwd_A = self.criterionIdt(self.fake_B, self.real_A)
            loss_fwd_B = self.criterionIdt(self.fake_A, self.real_B)
        else:
            loss_fwd_A, loss_fwd_B = 0, 0

        # combined loss
        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + \
                 (self.loss_cycle[self.DA] + self.loss_cycle[self.DB]) * self.lambda_cyc + \
                 (loss_idt_A + loss_idt_B) * self.lambda_idt + \
                 (loss_enc_A + loss_enc_B) * self.lambda_enc + \
                 (loss_fwd_A + loss_fwd_B) * self.lambda_fwd
        loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # G_A and G_B
        self.netG.zero_grads(self.DA, self.DB)
        self.backward_G()
        self.netG.step_grads(self.DA, self.DB)
        # D_A and D_B
        self.netD.zero_grads(self.DA, self.DB)
        self.backward_D()
        self.netD.step_grads(self.DA, self.DB)

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.data[0]) for i in l]
        D_losses, G_losses, cyc_losses = extract(self.loss_D), extract(self.loss_G), extract(self.loss_cycle)
        return OrderedDict([('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses)])

    def get_current_visuals(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        images = [util.tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_hyperparams(self, curr_iter):
        if curr_iter > self.opt.niter:
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            self.netG.update_lr(new_lr)
            self.netD.update_lr(new_lr)
            print('updated learning rate: %f' % new_lr)

        if self.opt.lambda_latent > 0:
            decay_frac = curr_iter / (self.opt.niter + self.opt.niter_decay)
            self.lambda_enc = self.opt.lambda_latent * decay_frac
