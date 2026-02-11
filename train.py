import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from dataloader import *
from utils import *
from model import *
from loss import *

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(args):

    # Choose the CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    # Create Dataset
    train_dataset = shapenet4096('train')

    # Create dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size = args.L_batch_size,
                                  shuffle=True,
                                  num_workers=args.E_workers,
                                  pin_memory=True)

    # Create Model
    Network = Network_Whole().cuda()
    Network.train()

    # Load Model if checkpoint is not none
    if args.E_ckpt_path != '':
        Network.load_state_dict(torch.load(args.E_ckpt_path))
        print('Load model successfully: %s' % (args.E_ckpt_path))

    # Create Loss Function
    loss_func = loss_whole().cuda()

    # Create Optimizer
    optimizer = optim.Adam(Network.parameters(), lr = 0.001, betas = (0.9, 0.999))

    # Training Processing
    color = generate_ncolors(args.N_num_cubes)
    num_batch = len(train_dataset)/args.L_batch_size
    batch_count = 0
    # train_loss_list = []
    # test_loss_list = []

    for epoch in range(args.L_epochs):
        for i, data in enumerate(train_dataloader, 0):
            points, normals, _, _, _ = data
            points, normals = points.cuda(), normals.cuda()
            optimizer.zero_grad()
            outdict = Network(pc = points)
            loss, loss_dict = loss_func(points, normals, outdict, None)
            loss.backward()
            optimizer.step()
            #print('Epoch', epoch, ': [', str(i) , '/', num_batch, ']')
            print_text(loss_dict, '', is_train = True, epoch = epoch, i = i, num_batch = num_batch, lr = 6e-4, print_freq_iter = 10)
            batch_count += 1

            with torch.no_grad():
                visualize_segmentation(points, color, outdict['assign_matrix'], '', 0, None)

        torch.save(Network.state_dict(), 'weights.pth')
        Network.train()

        #train_loss_list.append(loss.item())

    #return train_loss_list, test_loss_list

if __name__ == "__main__":

  class Data:
      def __init__(self, ):

        self.L_epochs = 500
        self.L_batch_size = 2
        self.E_workers = 0
        self.E_ckpt_path = ''
        self.N_num_cubes = 16

  args = Data()
  main(args)