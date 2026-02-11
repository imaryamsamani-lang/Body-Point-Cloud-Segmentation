import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import *
from utils import *
from model import *
from loss import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    if not os.path.exists('infer/'):
        os.makedirs('infer/')

    # Create Model
    Network = Network_Whole().cuda()
    Network.eval()

    # Load Model
    Network.load_state_dict(torch.load(args.checkpoint))
    print('Load model successfully: %s' % (args.checkpoint))

    color = generate_ncolors(args.N_num_cubes)

    # Create Dataset
    batch_size = 1

    cur_dataset = shapenet4096('test')

    cur_dataloader = DataLoader(cur_dataset,
                                batch_size = batch_size,
                                shuffle=False,
                                num_workers= args.E_workers,
                                pin_memory=True)
    
    infer(cur_dataloader, Network, color)

def infer(cur_dataloader, Network, color):

    save_path = 'infer/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for j, data in enumerate(cur_dataloader, 0):
        with torch.no_grad():
            points, normals, _, _, names = data
            points, normals = points.cuda(), normals.cuda()
            outdict = Network(pc = points)

            visualize_segmentation(points, color, outdict['assign_matrix'], save_path, _, names)
            if j==10:
                break

if __name__ == "__main__":
  class Data:
      def __init__(self):
        
        self.E_workers = 0
        self.checkpoint = r'weights.pth'
        self.N_num_cubes = 16

  args = Data()
  main(args)