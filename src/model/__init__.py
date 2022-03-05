import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        self.n_colors = args.n_colors

        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        print(self.model, file=ckp.log_file)


    def forward_(self, x):
        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            forward_function = self.model.forward
            if self.self_ensemble:
                if self.chop:
                    if isinstance(x,list):
                        return self.forward_chop_mul(x)
                    forward_function = self.forward_chop
                else:
                    forward_function = self.model.forward

                return self.forward_x8(x, forward_function)
            elif self.chop:
                if isinstance(x,list):
                    return self.forward_chop_mul(x)
                return self.forward_chop(x)
            else:
                return forward_function(x)

    def forward(self, x):
        if self.training:
            if self.n_GPUs > 1:
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                return self.model(x)
        else:
            if self.chop:
                if isinstance(x,list):
                    return self.forward_chop_mul(x)
                forward_function = self.forward_chop
            else:
                forward_function = self.model.forward

            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)


    def save(self, apath, epoch, is_best=False):
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        
        if resume == -2:
            load_from = torch.load(
                os.path.join(apath, 'model_best.pt'),
                **kwargs
            )
        elif resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    def forward_chop(self, x, shave=10, min_size=10000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        #############################################
        # adaptive shave
        # corresponding to scaling factor of the downscaling and upscaling modules in the network
        shave_scale = 4
        # max shave size
        shave_size_max = 12
        # get half size of the hight and width
        h_half, w_half = h // 2, w // 2
        # mod
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        # ditermine midsize along height and width directions
        h_size = mod_h * shave_scale + shave_size_max
        w_size = mod_w * shave_scale + shave_size_max
        #h_size, w_size = h_half + shave, w_half + shave
        ###############################################
        #h_size, w_size = adaptive_shave(h, w)
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]

        # print(f'===================={w_size} * {h_size}  {min_size}: forward_chop')
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            # print('===================Again: forward_chop')
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        c = self.n_colors
        output = Variable(x.data.new(b, c, h, w), volatile=True)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_chop_mul(self, x, shave=10, min_size=10000000):
        """
        multi-input
        """
        # print('***********************************forward once***************************')
        # print(x[0].size())
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x[0].size()
        ref=x[1]
        input=x[0]

        #############################################
        # adaptive shave
        # corresponding to scaling factor of the downscaling and upscaling modules in the network
        shave_scale = 4
        # max shave size
        shave_size_max = 12
        # get half size of the hight and width
        h_half, w_half = h // 2, w // 2
        # mod
        mod_h, mod_w = h_half // shave_scale, w_half // shave_scale
        # ditermine midsize along height and width directions
        h_size = mod_h * shave_scale + shave_size_max
        w_size = mod_w * shave_scale + shave_size_max
        #h_size, w_size = h_half + shave, w_half + shave
        ###############################################
        #h_size, w_size = adaptive_shave(h, w)
        lr_list = [
            input[:, :, 0:h_size, 0:w_size],
            input[:, :, 0:h_size, (w - w_size):w],
            input[:, :, (h - h_size):h, 0:w_size],
            input[:, :, (h - h_size):h, (w - w_size):w]]
        ref_list = [
            ref[:, :, 0:h_size, 0:w_size],
            ref[:, :, 0:h_size, (w - w_size):w],
            ref[:, :, (h - h_size):h, 0:w_size],
            ref[:, :, (h - h_size):h, (w - w_size):w]]
        if len(x)==3:
            three=x[2]
            three_list = [
                three[:, :, 0:h_size, 0:w_size],
                three[:, :, 0:h_size, (w - w_size):w],
                three[:, :, (h - h_size):h, 0:w_size],
                three[:, :, (h - h_size):h, (w - w_size):w]]
        # print(f'===================={w_size} * {h_size}  {min_size}: forward_chop_mul')
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                ref_batch = torch.cat(ref_list[i:(i + n_GPUs)], dim=0)
                if len(x)==3:
                    three_batch = torch.cat(three_list[i:(i + n_GPUs)], dim=0)
                    sr_batch = self.model([lr_batch,ref_batch,three_batch])
                else:
                    sr_batch = self.model([lr_batch,ref_batch])
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            # print('===================Again: forward_chop_mul')
            sr_list = []
            for i in range(0,len(lr_list)):
                if len(x)==3:
                    sr_list.append(self.forward_chop_mul([lr_list[i],ref_list[i],three_list[i]], shave=shave, min_size=min_size))
                else:
                    sr_list.append(self.forward_chop_mul([lr_list[i],ref_list[i]], shave=shave, min_size=min_size))

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        c = self.n_colors
        output = Variable(input.data.new(b, c, h, w), volatile=True)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()
            # print(op)
            # import pdb; pdb.set_trace()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        
        for a in args:
            x = [a]
            # import pdb; pdb.set_trace()
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y

