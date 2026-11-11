import random

import cv2
import torchvision.transforms as transforms

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.istrain = opt.isTrain
        self.Noise_paths = sorted(make_dataset(opt.data_root_noise))
        if self.istrain:
            self.Clean_paths = sorted(make_dataset(opt.data_root_clean))

    def __getitem__(self, index):
        if self.istrain:
            Noise_path = self.Noise_paths[index]
            Noise = cv2.imread(Noise_path)

            Clean_path = self.Clean_paths[index]
            Clean = cv2.imread(Clean_path)

            h, w, _ = Noise.shape
            width = h // 4
            h_off = random.randint(0, h - width)
            w_off = random.randint(0, w - width)
            Noise = Noise[h_off:h_off + width, w_off:w_off + width]
            Clean = Clean[h_off:h_off + width, w_off:w_off + width]
            Noise = cv2.cvtColor(Noise, cv2.COLOR_BGR2GRAY)
            Clean = cv2.cvtColor(Clean, cv2.COLOR_BGR2GRAY)

            rr = random.randint(0, 3)
            if rr == 1:
                Noise = cv2.flip(Noise, 0)
                Clean = cv2.flip(Clean, 0)
            elif rr == 2:
                Noise = cv2.flip(Noise, 1)
                Clean = cv2.flip(Clean, 1)
            elif rr == 3:
                Noise = cv2.flip(Noise, -1)
                Clean = cv2.flip(Clean, -1)
            else:
                pass

            Noise = transforms.ToTensor()(Noise)
            Clean = transforms.ToTensor()(Clean)


            return {'Noise': Noise, 'Clean': Clean, 'Noise_paths': Noise_path, 'Clean_paths': Clean_path}
        else:
            Noise_path = self.Noise_paths[index]
            Noise = cv2.imread(Noise_path)

            h, w, _ = Noise.shape

            Noise = cv2.cvtColor(Noise, cv2.COLOR_BGR2GRAY)

            Noise = transforms.ToTensor()(Noise)

            return {'Noise': Noise, 'Noise_paths': Noise_path}


    def __len__(self):
        return len(self.Noise_paths)

    def name(self):
        return 'AlignedDataset'