import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import Data
import json
from dataset import EcgDataset
from model import LSTMModel, CNNModel
from evaluation import evaluation

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_file = open("parameters.json")
    parameters = json.load(json_file)
    json_file.close()
    net = CNNModel(1, 10)
    optimizer = torch.optim.Adam(net.parameters(), lr = parameters["lr"])
    criterion = nn.BCELoss()

    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count())).cuda()
    cudnn.benchmark = True
    ecg_dataset = EcgDataset(is_train = True)
    train_loader = torch.utils.data.DataLoader(dataset = ecg_dataset, batch_size = 10)
    for epoch in range(parameters["num_epochs"]):
        net.train()
        for i, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            output = net(data)
            optimizer.zero_grad()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, parameters["num_epochs"], loss.item()))
        evaluation(net)

if __name__ == "__main__":
    train()
