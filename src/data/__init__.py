from importlib import import_module
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset

# This is a simple wrapper function for ConcatDataset
# class MyConcatDataset(ConcatDataset):
#     def __init__(self, datasets):
#         super(MyConcatDataset, self).__init__(datasets)
#         self.train = datasets[0].train

class Data:
    def __init__(self, args):
        self.loader_train = None
        if not args.test_only:

            # Maybe more training datasets will used in future work.
            datasets = []
            for d in args.data_train:
                module_name = d
                m = import_module('data.' + module_name.lower()) # 动态导入
                datasets.append(getattr(m, module_name)(args, name=d))

            self.loader_train = DataLoader(
                ConcatDataset(datasets),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads
            )
            

        self.loader_test = []

        # import pdb; pdb.set_trace()
        test_quality = eval(args.test_quality) if isinstance(args.test_quality, str) else args.test_quality
        for d in args.data_test:
            for q in test_quality:
                # testing by single image jpeg AR model
                if d in ['CUFED', 'Sun80', 'Set5', 'BSDS20']:
                    m = import_module('data.benchmark')
                    testset = getattr(m, 'Benchmark')(args, train=False, name=d, quality=q)
                # testing by two image jpeg AR model
                elif d in ['CUFED_REF', 'Set5_REF', 'BSDS20_REF',\
                        'Sun80_REF']:
                    module_name = d
                    if args.n_input == 2:
                        m = import_module('data.refimgs')
                    testset = getattr(m, 'RefImgs')(args, train=False, name=d, quality=q)
                else:
                    module_name = d
                    m = import_module('data.' + module_name.lower())
                    testset = getattr(m, module_name)(args, train=False, name=d, quality=q)
                
                self.loader_test.append(DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not args.cpu,
                    num_workers=args.n_threads
                ))
