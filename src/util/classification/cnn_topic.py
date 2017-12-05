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

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 1, (4, self.vec_size))
        self.mp1 = nn.MaxPool2d((self.num_vec-4, 1))
        self.conv2 = nn.Conv2d(1, 1, (3, self.vec_size))
        self.mp2 = nn.MaxPool2d((self.num_vec-3, 1))
        self.conv3 = nn.Conv2d(1, 1, (2, self.vec_size))
        self.mp3 = nn.MaxPool2d((self.num_vec-2, 1))
        self.fc1 = nn.Linear(3, self.num_classes)
        self.softmax = nn.Softmax()

        self.loss_function = nn.MSELoss().cuda()
        self.optimizer = optim.Adadelta(self.parameters())

    def set_training_data(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def forward(self, doc):
        x = doc
        if len(x.size()) == 2:
            (H, W) = x.data.size()
            x = x.view(1, 1, H, W)
        x11 = self.conv1(x)
        x11 = self.relu(x11)
        x11 = self.mp1(x11)
        # x12 = self.conv1(x)
        # x12 = self.relu(x12)
        # x12 = self.mp1(x12)
        x21 = self.conv2(x)
        x21 = self.relu(x21)
        x21 = self.mp2(x21)
        # x22 = self.conv2(x)
        # x22 = self.relu(x22)
        # x22 = self.mp2(x22)
        x31 = self.conv3(x)
        x31 = self.relu(x31)
        x31 = self.mp3(x31)
        # x32 = self.conv3(x)
        # x32 = self.relu(x32)
        # x32 = self.mp3(x32)
        x = torch.cat((x11,x21,x31), 2)
        x = x.view(1,1,3)
        x = self.fc1(x)
        x = x.view(1,self.num_classes)
        x = self.softmax(x)
        return x

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
    X_train, X_test, y_train, y_test = doc2vec_loader.load_data(max_num_vecs=300)
    nn = cnn.CnnTopic(num_vec=300)
    nn.cuda()

    for idx, x in enumerate(X_test):
        x = Variable(torch.FloatTensor(x).cuda())

        print(nn.forward(x))
        return

if __name__ == "__main__":
    main()
