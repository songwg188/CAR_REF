import argparse
import template
import random

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--trainer', default='trainer2',
                    help='You can set trainer')  
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=16,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../dataset',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-200/801-810',
                    help='train/test data range')
parser.add_argument("--quality", default=[1,60], type=str, 
                    help="the range of quality factors when training")
parser.add_argument("--test_quality", default=[10, 20, 30, 40, 50], type=str, 
                    help="the range of quality factors when testing")
parser.add_argument('--scale', default='1',
                    help='super resolution scale')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')


# ------------------------------------------------------ #
# patchsize = 192

parser.add_argument('--patch_size', type=int, default=80,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--scale_aug', action='store_true',
                    help='use multi-scale data augmentation')

# Model specifications
parser.add_argument('--model', default='qmar',
                    help='model name')
parser.add_argument('--qm', action='store_true',
                    help='use qm as conducted in Jianweis work')
parser.add_argument('--qv', action='store_true',
                    help='use qv in my new work')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_coarseblocks', type=int, default=8,
                    help='number of residual blocks in the coarse branch of each CFB')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--n_convgroup', type=int, default=1,
                    help='number of groups of each conv')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')
parser.add_argument('--visdom', action='store_true',
                    help='show log with visdom')
parser.add_argument('--onlydraw', action='store_true',
                    help='only show training lines in visdom and then exit the program')

# setting of reference image
parser.add_argument('--mul_input', action='store_true',
                    help='multi input')
parser.add_argument('--n_input', type=int, default=2,
                    help='num of input image')
parser.add_argument('--ref_level', type=str, default='1',
                    help='similarity level of refimg and input image in test dataset of cufed, range in 1-5')
# qm in ref and input is diff, if q=100, the num in qm are equal to 1 
parser.add_argument('--qm_diff', action='store_true',
                    help='multi input')

# set of rdn
parser.add_argument('--RDNconfig', type=str, default='A',
                    help='RDNconfig')

args = parser.parse_args()
template.set_template(args)

args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

# quality = eval(args.quality) if isinstance(args.quality, str) else args.quality
# if len(quality) == 2:
#     assert quality[0] <= quality[1], f'Illegal quality range {quality}'
#     quality = random.randint(quality[0], quality[1])
# else:
#     quality = quality[0]
# args.quality = quality
args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1000

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

