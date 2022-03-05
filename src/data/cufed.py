# low quality image and reference img in training dataset cufed 
# fixed 20210710
import os
from data import data_ref

class CUFED(data_ref.Data): # TODO train test data
    def __init__(self, args, name='CUFED', train=True, benchmark=False):
        super(CUFED, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        return super(CUFED, self)._scan()

    def _set_filesystem(self, dir_data):
        super(CUFED, self)._set_filesystem(dir_data)

