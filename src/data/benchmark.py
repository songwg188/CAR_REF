import os
import glob
from data import common
from data import data

class Benchmark(data.Data):
    def __init__(self, args, name='', train=True, benchmark=True, quality=10):
        super(Benchmark, self).__init__(
            args, name=name, train=train, benchmark=True
        )
        self.quality = [quality]

    def _scan(self):
        names_hr = super(Benchmark, self)._scan()
        # testing by single image jpeg AR model
        if self.name == 'CUFED':
            names_hr = sorted(glob.glob(os.path.join(self.dir_hr, '*_0' + self.ext)))
        return names_hr
    
    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'ar_benchmark', self.name)
        if self.name == 'CUFED':
            self.dir_hr = os.path.join(dir_data, self.name,'test/CUFED5')
            self.ext = '.png'
        if self.name == 'Sun80':
            self.dir_hr = os.path.join(dir_data, self.name,'Sun_Hays_SR_groundtruth')
            self.ext = '.jpg'
        if self.name == 'Set5':
            self.dir_hr = os.path.join(self.apath,'HR')
            self.ext = '.png'
        if self.name == 'BSDS20':
            self.dir_hr = os.path.join(self.apath, 'HR')
            self.ext = '.png'



