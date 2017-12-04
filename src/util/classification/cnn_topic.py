from torch.autograd import Variable
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

class CnnTopic(nn.Module):
    def __init__(self, num_vec=100):
        super(CnnTopic, self).__init__()
        # Intialize NeuralNetwork
        self.vec_size = 300
        self.num_vec = num_vec
        self.num_classes = 10

        self.X_train = None
        self.y_train = None

        in_layer = self.vec_size*self.num_vec
        out_layer = self.num_classes
        self.linear = nn.Linear(in_layer, out_layer).cuda()
        self.softmax = nn.Softmax().cuda()
        self.loss_function = nn.MSELoss().cuda()
        self.optimizer = optim.Adadelta(self.parameters())

    def set_training_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def forward(self, doc):
        (H, W) = doc.data.size()
        doc = doc.view(1, H*W)
        doc = self.linear(doc)
        doc = F.sigmoid(doc)
        doc = self.softmax(doc)
        return doc

    def train(self):
        if self.X_train == None or self.y_train == None:
            print("Please specify training data by set_training_data()")

        # training
        epoch = 1
        if epoch > 1:
            print("== Start training for {0:d} epochs".format(epoch))
        for i in range(epoch):
            # batch_idx = 0
            # for batch_idx, (x, target) in enumerate(train_loader):
            for idx, x in enumerate(self.X_train):
                self.optimizer.zero_grad()

                x = torch.FloatTensor(x).cuda()
                target = self.ToOneHot(self.y_train[idx])

                x, target = Variable(x), Variable(target)
                x_pred = self.forward(x)
                loss = self.loss_function(x_pred, target)
                loss.backward()
                self.optimizer.step()
            #     if (batch_idx+1)% 100 == 0:
            #         # print '==>>> batch index: {}, train loss: {:.6f}'.format(batch_idx, loss.data[0])
            #         print '==>>> batch index: {}'.format(batch_idx+1)
            # print '==>>> batch index: {}'.format(batch_idx+1)
            if epoch > 1:
                print("-- Finish epoch {0:d}".format(i+1))

    def ToOneHot(self, target):
        # oneHot encoding
        label = []
        label .append([1 if i==target else 0 for i in range(self.num_classes)])
        return torch.FloatTensor(label).cuda()

    def FromOneHot(self, target):
        # oneHot encoding to class id
        values, indices = target.max(0)
        return indices[0]

def main():
    pass

if __name__ == "__main__":
    main()
