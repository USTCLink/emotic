import torch
from model import Net
from dataset import Emotic
import torch.optim as optim
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataloader = DataLoader(Emotic('train'), batch_size=52, shuffle=True)

net = Net().to(device)

image_feature_extraction_pretrained_model_file = 'resnet50_places365.pth.tar'
checkpoint = torch.load(image_feature_extraction_pretrained_model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
net.image_feature_extraction.load_state_dict(state_dict)

for param in net.body_feature_extraction.parameters():
    param.requires_grad = False
for param in net.image_feature_extraction.parameters():
    param.requires_grad = False
for param in net.image_feature_extraction.fc.parameters():
    param.requires_grad = True

params = list(net.fc.parameters()) + list(net.image_feature_extraction.fc.parameters()) + list(net.fusion.parameters())
optimizer = optim.Adam(params, lr=1e-3)


def train():
    net.train()
    for idx, (image, body, label) in enumerate(dataloader):
        image, body, label = image.to(device), body.to(device), label.to(device)
        output = net(body, image)
        loss = torch.sum(torch.pow(output - label, 2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (idx + 1) % 25 == 0:
            print(epoch, idx + 1, loss.item())


if __name__ == '__main__':
    for epoch in range(1):
        train()
    torch.save(net.state_dict(), 'net.pth')


