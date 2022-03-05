import os
import glob
import numpy as np
from data import common
from data import data_ref

class RefImgs(data_ref.Data):
    def __init__(self, args, name='', train=True, benchmark=True, quality=10):
        self.ref_level = args.ref_level
        super(RefImgs, self).__init__(
            args, name=name, train=train, benchmark=True
        )
        self.quality = [quality]

    def _scan(self):
        names_hr_input, names_hr_ref = '', ''
        if self.name.find('_REF') >= 0:
            if self.name.find('CUFED') >= 0:
                self.apath = os.path.join(self.args.dir_data, self.name.split('_REF')[0])
                self.dir_hr_input = os.path.join(self.apath, 'test/CUFED5')
                self.dir_hr_ref = os.path.join(self.apath, 'test/CUFED5')
                self.ext = '.png'
                names_hr_input = sorted(glob.glob(os.path.join(self.dir_hr_input, '*_0' + self.ext)))
                if self.name.find('_INPUT') >= 0:
                    names_hr_ref = names_hr_input
                else:
                    names_hr_ref = sorted(glob.glob(os.path.join(self.dir_hr_ref, '*_' + self.ref_level + self.ext)))
                # print(len(names_hr_ref))

            if self.name.find('Sun80') >= 0:
                self.apath = os.path.join(self.args.dir_data, self.name.split('_REF')[0])
                self.dir_hr_input = os.path.join(self.apath, 'Sun_Hays_SR_groundtruth')
                self.dir_hr_ref = os.path.join(self.apath, 'Sun_Hays_SR_scenematches_sim')
                self.ext = '.jpg'
                names_hr_input = sorted(glob.glob(os.path.join(self.dir_hr_input, '*' + self.ext)))
                if self.name.find('_INPUT') >= 0:
                    names_hr_ref = names_hr_input
                else:
                    names_hr_ref_dir = sorted(glob.glob(os.path.join(self.dir_hr_ref, '*' + self.ext)))
                    names_hr_ref = list()
                    for d in names_hr_ref_dir:
                        name = sorted(glob.glob(os.path.join(d, '*' + self.ext)))[1]
                        names_hr_ref.append(name)

            
            if self.name.find('Set5') >= 0:
                self.apath = os.path.join(self.args.dir_data, 'ar_benchmark', self.name.split('_REF')[0])
                self.dir_hr_input = os.path.join(self.apath, 'HR')
                self.dir_hr_ref = os.path.join(self.apath, 'HR_sim')
                self.ext = '.png'
                names_hr_input = sorted(glob.glob(os.path.join(self.dir_hr_input, '*' + self.ext)))
                if self.name.find('_INPUT') >= 0:
                    names_hr_ref = names_hr_input
                else:
                    names_hr_ref_dir = sorted(glob.glob(os.path.join(self.dir_hr_ref, '*' + self.ext)))
                    names_hr_ref = list()
                    for d in names_hr_ref_dir:
                        name = sorted(glob.glob(os.path.join(d, '*' + 'jpg')))[0] #取第一个作为参考图像
                        names_hr_ref.append(name)        

            if self.name.find('BSDS20') >= 0:
                self.apath = os.path.join(self.args.dir_data, 'ar_benchmark', self.name.split('_REF')[0])
                self.dir_hr_input = os.path.join(self.apath, 'HR')
                self.dir_hr_ref = os.path.join(self.apath, 'HR_sim')
                self.ext = '.png'
                names_hr_input = sorted(glob.glob(os.path.join(self.dir_hr_input, '*' + self.ext)))
                # print(names_hr_input)
                if self.name.find('_INPUT') >= 0:
                    names_hr_ref = names_hr_input
                else:
                    # names_hr_ref_dir = sorted(glob.glob(os.path.join(self.dir_hr_ref, '*' + self.ext)))
                    names_hr_ref = list()
                    for d in names_hr_input:
                        im_name= d.split('/')[-1]
                        #print(d)
                        name = sorted(glob.glob(os.path.join(self.dir_hr_ref, im_name, '*' + 'jpg')))[0] #取第一个作为参考图像
                        #print(name)
                        names_hr_ref.append(name)         


            # print(names_hr_input)
            # print(names_hr_ref)
            return names_hr_input, names_hr_ref



    def _set_filesystem(self, dir_data): 
        pass
