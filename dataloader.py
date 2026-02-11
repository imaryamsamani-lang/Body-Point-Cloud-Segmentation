import os
import numpy as np
import torch
import torch.utils.data as data
import random

class shapenet4096(data.Dataset):
    def __init__(self, phase, ):
        super().__init__()

        if phase=='train':
            self.data_list = os.listdir('NPY')[:500]
        else:
            self.data_list = os.listdir('NPY')[500:]

        self.indexes = [item.split('.')[0].split('combined_data')[-1] for item in self.data_list]

    def __getitem__(self, idx):
        idx = random.choice(self.indexes)
        
        cur_name = str(idx)
        cur_data = torch.from_numpy(np.load("NPY/combined_data" + idx + ".npy")).float()
        cur_points = cur_data[:,0:3]
        cur_normals = cur_data[:,3:]
        cur_points_num = 4096
        cur_values = -1
        return cur_points, cur_normals, cur_points_num, cur_values, cur_name

    def __len__(self):
        return len(self.data_list)