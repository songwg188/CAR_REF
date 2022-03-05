import os
import math
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def model_complexity_info(model, in_channel, p_size):
    # show model informantion
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (in_channel, p_size, p_size), as_strings=True, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def model_complexity_info_mul_input(model, in_channel, p_size, device):
    # show model informantion
    # two input  fixed 20210710 
    def prepare_input(resolution):
        x1 = torch.FloatTensor(1, *resolution[0])
        x2 = torch.FloatTensor(1, *resolution[1])
        return dict(x = [x1.to(device), x2.to(device)])
    from ptflops import get_model_complexity_info
    input = (((in_channel, p_size,p_size),(in_channel,p_size,p_size)))
    flops, params = get_model_complexity_info(model, input,input_constructor=prepare_input, as_strings=True, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def model_complexity_info_three_input(model, in_channel, p_size,device):
    # show model informantion
    # two input  fixed 20210710 
    def prepare_input(resolution):
        x1 = torch.FloatTensor(1, *resolution[0])
        x2 = torch.FloatTensor(1, *resolution[1])
        x3 = torch.FloatTensor(1, *resolution[2])
        return dict(x = [x1.to(device), x2.to(device), x3.to(device)])
    from ptflops import get_model_complexity_info
    input = (((in_channel, p_size,p_size),(in_channel,p_size,p_size),(in_channel,p_size,p_size)))
    flops, params = get_model_complexity_info(model, input,input_constructor=prepare_input, as_strings=True, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args

        self.test_quality = eval(args.test_quality) if isinstance(args.test_quality, str) else args.test_quality

        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
                # print(self.log.size())
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)

        qualities = self.test_quality

        for idx_data, d in enumerate(self.args.data_test):
            label = 'AR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)

            for i, q in enumerate(qualities):
                plt.plot(
                    axis,
                    self.log[:, idx_data* len(qualities) + i].numpy(),
                    label='Quality {}'.format(q)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    imageio.imwrite(filename, tensor.numpy())
        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('HQ', 'LQ', 'GT')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

import torch.nn as nn
from torchvision import transforms
class DoG(nn.Module):
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0)):
        super(DoG, self).__init__()
        self.gaussian_blurring = transforms.GaussianBlur(kernel_size, sigma)

    def split(self, x):
        subband_l = self.gaussian_blurring(x)
        subband_h = x - subband_l
        return torch.cat((x, subband_l, subband_h), 0)

    def merge(self, x):
        x.size()


def seed_torch(seed=1):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def display_args(args):
    from prettytable import PrettyTable
    arg_s = vars(args)
    x = PrettyTable()
    n_column = 2
    
    text = []
    for i in range(n_column):
        text.append("Arg_name_{}".format(i))
        text.append("Value_{}".format(i))
    x.field_names = text
    texts = []
    
    for i, arg_name in enumerate(arg_s):
        texts.append(arg_name)
        texts.append(arg_s[arg_name])
        if (i + 1) % n_column == 0:
            x.add_row(texts)
            texts = []
    print(x)
    