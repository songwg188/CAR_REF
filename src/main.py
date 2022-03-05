# -*- coding: utf-8 -*
import skimage
import torch
import pdb
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import numpy as np
import sys

from importlib import import_module

# 设置随机种子
utility.seed_torch(args.seed)
utility.display_args(args)


# 选择一个训练器
module = import_module(args.trainer)
Trainer = getattr(module, 'Trainer')


# 配置检查点
checkpoint = utility.checkpoint(args)

if args.visdom:
    # set visualization
    from visdom import Visdom
    env_name = 'AR_' + args.save
    vis = Visdom(port=8097, server="http://localhost", env=env_name)
    if len(checkpoint.log) > 0:
        logger_test =vis.line(X=torch.arange(0, len(checkpoint.log)), Y=checkpoint.log[:,:], opts=dict(title="PSNR"))
    else:
        logger_test = vis.line(np.arange(50), opts=dict(title="PSNR"))
    if args.onlydraw:
        sys.exit() 
else:
    logger_test = None
    vis = None



if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)

    # show model informantion
    if args.n_input == 2:
        utility.model_complexity_info_mul_input(model, args.in_channel, args.patch_size, device = torch.device('cpu' if args.cpu else 'cuda'))
    elif args.n_input == 3:
        utility.model_complexity_info_three_input(model, args.in_channel, args.patch_size, device = torch.device('cpu' if args.cpu else 'cuda'))
    else:
        utility.model_complexity_info(model, args.in_channel, args.patch_size)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint, logger_test, vis)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

