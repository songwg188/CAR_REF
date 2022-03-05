# single image jpeg training in dataset cufed 
# fixed 20210715
import os
from data import data

class CUFED_SINGLE(data.Data): # TODO train test data
    def __init__(self, args, name='CUFED_SINGLE', train=True, benchmark=False):
        super(CUFED_SINGLE, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        return super(CUFED_SINGLE, self)._scan()

    def _set_filesystem(self, dir_data): 
        # set basic information of dataset
        self.apath = os.path.join(dir_data, self.name.split('_')[0])
        self.dir_hr = os.path.join(self.apath, 'train/input')
        self.ext = '.png'
        pass


