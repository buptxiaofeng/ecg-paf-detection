import torch
from dataset import EcgDataset

def evaluation(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecg_dataset = EcgDataset(is_train = False)
    test_loader = torch.utils.data.DataLoader(dataset = ecg_dataset)
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for data, label in test_loader:
            total = total + 1
            flag = True
            for i in range(6):
                data = data[:,:, i: (i+1) * 38400, :].to(device)
                label = label.to(device)
                output = model(data)
                predict = torch.argmax(output)
                label = torch.argmax(label)
                if predict == 1 and predict == label:
                    correct = correct + 1
                    flag = False
                    break
                if predict != label:
                    flag = False
            if flag:
                correct = correct + 1

        print('Test Accuracy of the model on the test :{} %'.format(100 * correct / total))
