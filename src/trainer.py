import os
import math
from decimal import Decimal

import utility
from utils import util_metric

import torch
import torch.nn.utils as utils
from tqdm import tqdm
import pdb
import numpy as np

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, logger_test, vis):
        self.args = args

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.logger_test = logger_test
        self.vis = vis

    def train(self):

        epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lq, gt, _) in enumerate(self.loader_train):

            lq, gt = self.prepare(lq, gt)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            hq = self.model(lq)

            loss = self.loss(hq, gt)

            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

            # del lq, gt, hq, loss
            # torch.cuda.empty_cache() # 为了防止别人抢资源，这里不手动释放

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')

        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            # pdb.set_trace()
            quality = d.dataset.quality[0]
            for lq, gt, LQimg, hw, filename in tqdm(d, ncols=80):
                lq, gt = self.prepare(lq, gt)
                hq = self.model(lq)
                hq = utility.quantize(hq, self.args.rgb_range)

                # cut 
                hq = hq[:, :, 0:hw[0], 0:hw[1]]
                LQimg = LQimg[:, :, 0:hw[0], 0:hw[1]]
                gt = gt[:, :, 0:hw[0], 0:hw[1]]

                save_list = [hq]
                self.ckp.log[-1, idx_data] += util_metric.calc_PSNR(hq, gt, self.args.rgb_range)
                if self.args.save_gt:
                    save_list.extend([LQimg, gt])

                if self.args.save_results:
                    # pdb.set_trace()
                    self.ckp.save_results(d, filename[0], save_list, quality)



            self.ckp.log[-1, idx_data] /= len(d)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                    d.dataset.name,
                    quality,
                    self.ckp.log[-1, idx_data],
                    best[0][idx_data],
                    best[1][idx_data]
                )
            )

        # del lq, gt, LQimg
        torch.cuda.empty_cache()  # 测试完之后可以释放一下，为下一轮训练腾空间

        # write visdom
        if self.logger_test:
            Y = self.ckp.log
            X = torch.arange(0, len(Y))
            self.vis.line(X=X, Y=Y, win=self.logger_test, opts=dict(
                legend=[d.dataset.name + "q" + str(d.dataset.quality[0]) for _, d in enumerate(self.loader_test)], title="PSNR"))


        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.epochs

