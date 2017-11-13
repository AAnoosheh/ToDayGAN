import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.n_domains = opt.n_domains
        self.DA, self.DB = None, None

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G (both G & F), D (both D_A & D_B)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG, self.n_domains, opt.norm, not opt.no_dropout, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, self.n_domains, opt.norm, opt.no_lsgan, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.downsample = torch.nn.AvgPool2d(3, stride=4)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # initialize loss storage
            self.loss_D, self.loss_G = [0]*self.n_domains, [0]*self.n_domains
            self.loss_cycle, self.loss_idt = [0]*self.n_domains, [0]*self.n_domains

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.DA = input['DA'][0]
        self.DB = input['DB'][0]
        self.image_paths = input['path']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        real = Variable(self.input_A, volatile=True)
        self.visuals = [real]
        self.labels = ['real_%d' % self.DA]

        for d in range(self.n_domains):
            if d != self.DA:
                fake = self.netG.forward(real, self.DA, d)
                rec = self.netG.forward(fake, d, self.DA)
                self.visuals += [fake, rec]
                self.labels += ['fake_%d' % d, 'rec_%d' % d]

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, real, fake, domain):
        # Real
        pred_real = self.netD.forward(real, domain=domain)
        loss_D_real = self.criterionGAN(pred_real, domain, True)
        # Fake
        pred_fake = self.netD.forward(fake.detach(), domain=domain)
        loss_D_fake = self.criterionGAN(pred_fake, domain, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        self.loss_D[self.DA] = self.backward_D_basic(self.real_B, fake_B, self.DB)

    def backward_D_B(self):
        fake_A = self.fake_pools[self.DA].query(self.fake_A)
        self.loss_D[self.DB] = self.backward_D_basic(self.real_A, fake_A, self.DA)

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_idt = self.opt.identity
        lambda_fwd = self.opt.lambda_forward

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG.forward(self.real_B, self.DA, self.DB)
            self.loss_idt[self.DA] = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG.forward(self.real_A, self.DB, self.DA)
            self.loss_idt[self.DB] = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt[self.DA] = 0
            self.loss_idt[self.DB] = 0

        # GAN loss
        # D_A(G_A(A))
        self.fake_B = self.netG.forward(self.real_A, self.DA, self.DB)
        pred_fake = self.netD.forward(self.fake_B, self.DB)
        self.loss_G[self.DA] = self.criterionGAN(pred_fake, self.DB, True)
        # D_B(G_B(B))
        self.fake_A = self.netG.forward(self.real_B, self.DB, self.DA)
        pred_fake = self.netD.forward(self.fake_A, self.DA)
        self.loss_G[self.DB] = self.criterionGAN(pred_fake, self.DA, True)

        # Forward cycle loss
        self.rec_A = self.netG.forward(self.fake_B, self.DB, self.DA)
        self.loss_cycle[self.DA] = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.rec_B = self.netG.forward(self.fake_A, self.DA, self.DB)
        self.loss_cycle[self.DB] = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # Optional loss on downsampled image before and after
        if lambda_fwd > 0:
            loss_fwd_A = self.criterionIdt(self.downsample(self.fake_B), self.downsample(self.real_A)) * lambda_fwd
            loss_fwd_B = self.criterionIdt(self.downsample(self.fake_A), self.downsample(self.real_B)) * lambda_fwd
        else:
            loss_fwd_A, loss_fwd_B = 0, 0

        # combined loss
        loss_G = self.loss_G[self.DA] + self.loss_G[self.DB] + self.loss_cycle[self.DA] + self.loss_cycle[self.DB] + \
                 self.loss_idt[self.DA] + self.loss_idt[self.DB] + loss_fwd_A + loss_fwd_B
        loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.optimizer_D.step()
        # D_B
        self.optimizer_D.zero_grad()
        self.backward_D_B()
        self.optimizer_D.step()

    def get_current_errors(self):
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.data[0]) for i in l]
        D_losses = extract(self.loss_D)
        G_losses = extract(self.loss_G)
        cyc_losses = extract(self.loss_cycle)
        if self.opt.identity > 0.0:
            idt_losses = extract(self.loss_idt)
            return OrderedDict([('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses), ('idt', idt_losses)])
        else:
            return OrderedDict([('D', D_losses), ('G', G_losses), ('Cyc', cyc_losses)])

    def get_current_visuals(self, testing=False):
        if not testing:
            self.visuals = [self.real_A, self.fake_B, self.rec_A, self.real_B, self.fake_A, self.rec_B]
            self.labels = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
            if self.opt.identity > 0.0:
                self.visuals += [self.idt_A, self.idt_B]
                self.labels += ['idt_A', 'idt_B']

        images = [util.tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
