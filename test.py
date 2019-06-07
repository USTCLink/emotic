import torch
from model import Net
from dataset import Emotic
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataloader = DataLoader(Emotic('test'))

net = Net().to(device)


def test():
    net.eval()
    net.load_state_dict(torch.load('net.pth'))
    for idx, (image, body, label) in enumerate(dataloader):
        image, body, label = image.to(device), body.to(device), label.to(device)
        output = net(body, image)
        output = output > 0.5
        label = label.type_as(output)
        overlap = torch.sum(output * label).item()
        union = 26 - torch.sum((1 - output) * (1 - label)).item()
        jaccard_coefficient = overlap / union
        jaccard_coefficient_list.append(jaccard_coefficient)


if __name__ == '__main__':
    jaccard_coefficient_list = []
    test()
    fig, chart = plt.subplots(nrows=1, figsize=(9, 6))
    chart.hist(jaccard_coefficient_list, 100, density=1, histtype='bar')
    chart.set_title('frequency histogram')
    fig.subplots_adjust(hspace=0.4)
    plt.show()
