from __future__ import print_function
import os
import math
import torch
from torch import np
import argparse
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets 
from torchvision import utils 
import torch.utils.data as data_utils
from torch.utils.serialization import load_lua

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 40)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--gpu_id', default='2', type=str, 
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--dataPath', default='/home/hankuan1993/dataset/stl10', type=str, 
                    help='where to load / save dataset')
parser.add_argument('--latSize', default=1024, type=str, metavar='N',
                    help='the dimension of latent variable')
parser.add_argument('--notLoadPrev', action='store_true', default=False,
                    help='load previously saved checkpoint')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.loadPrev = not args.notLoadPrev


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 3,1,1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256,256, 3,1,1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256,512, 3,1,1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512,512, 3,1,1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512,1024, 3,1,1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.fc11 = nn.Linear(4*4*1024, args.latSize)
        self.fc12 = nn.Linear(4*4*1024, args.latSize)
        self.fc2 = nn.Linear(args.latSize, 4*4*1024)
        self.bn6 = nn.BatchNorm2d(1024)
        # nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)
        self.conv6 = nn.ConvTranspose2d( 1024,512, 4, 2, 1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv7 = nn.ConvTranspose2d( 512,512, 4, 2, 1)
        self.bn8 = nn.BatchNorm2d(512)
        self.conv8 = nn.ConvTranspose2d( 512,256, 4, 2, 1)
        self.bn9 = nn.BatchNorm2d(256)
        self.conv9 = nn.ConvTranspose2d( 256,256, 4, 2, 1)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv10 = nn.ConvTranspose2d( 256,3, 4, 2, 1)
        
    def encoder(self, x):
        h1 = F.relu(self.bn1(F.max_pool2d(self.conv1(x),2)))
        h2 = F.relu(self.bn2(F.max_pool2d(self.conv2(h1),2)))
        h3 = F.relu(self.bn3(F.max_pool2d(self.conv3(h2),2)))
        h4 = F.relu(self.bn4(F.max_pool2d(self.conv4(h3),2)))
        h5 = F.relu(self.bn5(F.max_pool2d(self.conv5(h4),2)))
        fc = h5.view(-1, 4*4*1024)
        lcm = F.relu(self.fc11(fc))
        lcv = F.relu(self.fc12(fc))
        return lcm, lcv
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decoder(self, z):
        h6 = F.relu(self.bn6(self.fc2(z).view(-1,1024,4,4)))
        # do not upsample, instead, use spatial fulll convolution. 
        # h6us = F.upsample_nearest(h6,scale_factor = 2)
        h7 = F.relu(self.bn7(self.conv6(h6)))
        h8 = F.relu(self.bn8(self.conv7(h7)))
        h9 = F.relu(self.bn9(self.conv8(h8)))
        h10 = F.relu(self.bn10(self.conv9(h9)))
        h11 = F.sigmoid(self.conv10(h10))
        return h11
    
    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1,3,128,128))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(xRecon, x, mu, logvar):

    reconstruction_function = nn.BCELoss()
    reconstruction_function.size_average = False
    BCE = reconstruction_function(xRecon, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD


def main():
    # cofiguration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id   
    torch.manual_seed(args.seed)
    if args.cuda:
        print('using CUDA with gpu_id:')
        print(args.gpu_id)
        torch.cuda.manual_seed(args.seed)
    # dataset
    unlabeledset = load_lua('/home/hankuan1993/dataset/stl10/stl10-unlabeled-scaled-tensor.t7')
    unlabeledsetnp = unlabeledset.numpy()
    trainset = load_lua('/home/hankuan1993/dataset/stl10/stl10-train-scaled-tensor.t7')
    trainsetnp = trainset.numpy()
    testset = load_lua('/home/hankuan1993/dataset/stl10/stl10-test-scaled-tensor.t7')
    testsetnp = testset.numpy()
    trainsetnp = np.concatenate((unlabeledsetnp, trainsetnp),axis = 0)
    trainlen = len(trainsetnp)
    testlen = len(testsetnp)
    # model
    model = CVAE()
    if args.loadPrev:
        print('====> loading previously saved model ...')
        model.load_state_dict(torch.load('./cvaeCheckPoint.pth'))           
    if args.cuda:
        print('====> loading model to gpu ...')
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # train and evaluate
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        train_loss = 0
        traintime = math.ceil(trainlen / args.batch_size)
        shuffleidx = np.random.permutation(trainsetnp.shape[0])
        for batch_idx in range(traintime):
            datanp = trainsetnp[shuffleidx[batch_idx*args.batch_size : (batch_idx+1) * args.batch_size],:,:,:].astype(np.float32) / 255.0
            data = torch.from_numpy(datanp)
            data = Variable(data)
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            optimizer.step()
#            if batch_idx % args.log_interval == 0:
#                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                    epoch, batch_idx * len(data), len(train_loader.dataset),
#                    100. * batch_idx / len(train_loader),
#                    loss.data[0] / len(data)))
        print('====> Epoch: {} Train Average loss: {:.4f}'.format(
              epoch, train_loss / traintime / args.batch_size))
        
        # evaluate
        model.eval()        
        test_loss = 0
        for test_idx in range(testlen):
            datanp = testsetnp[test_idx,:,:,:].astype(np.float32) / 255.0
            data = torch.from_numpy(datanp)
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

        test_loss /= testlen
        print('====> Test set loss: {:.4f}'.format(test_loss))
    # Since training is finished, save it! :)    
    torch.save(model.state_dict(), './cvaeCheckPoint.pth')

        
def test():
    # cofiguration
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id   
    torch.manual_seed(args.seed)
    # model
    model = CVAE()
    model.load_state_dict(torch.load('./cvaeCheckPoint.pth'))    
    # , map_location=lambda storage, location: 'cpu'
    model.cpu()
    model.eval()

    testset = load_lua('/home/hankuan1993/dataset/stl10/stl10-test-scaled-tensor.t7')
    testsetnp = testset.numpy() 
    datanp = testsetnp[21:30,:,:,:].astype(np.float32) / 255.0
    data = torch.from_numpy(datanp)
    data = Variable(data)
    recon_batch, mu, logvar = model(data)
    recon_batchnp = recon_batch.data.numpy()
    print(len(recon_batchnp))
    utils.save_image(data.data,'cvae_orginal_result.jpg')
    utils.save_image(recon_batch.data,'cvae_recons_result.jpg')
        
if __name__ == '__main__':
    args.cuda = 0
    test()   
    

        
        
        
