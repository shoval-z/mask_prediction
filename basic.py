from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from dataset import mask_dataset


class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=False)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 3))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))

    def forward(self, x):
        x = self.features1(x) # x =[batch_size, 128, 38, 38]
        x = self.features2(x) #x = [batch_size, 512, 10, 10]
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x) # x=[batch_size, 512, 1, 1]
        x = x.view(x.shape[0], -1) #x = [batch_size, 512]
        classifier = self.classifier(x) #[batch_size, num_class]
        bb = self.bb(x) #[batch_size, 4]
        return classifier, bb

def val_metrics(model, device, valid_dl, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y_class, y_bb in valid_dl:
        batch = y_class.shape[0]
        x = x.to(device).float()
        y_class = y_class.to(device)
        y_bb = y_bb.to(device).float()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class.squeeze(1), reduction="sum")
        loss_bb = F.l1_loss(out_bb, y_bb.squeeze(1), reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb/C
        _, pred = torch.max(out_class, 1)
        pred = (pred == 2).float()
        y_class = (y_class == 2).float().squeeze(1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    return sum_loss/total, correct/total


def train_epocs(model,device, optimizer, train_dl, test_dl, epochs=10,C=1000):
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        accuracy =0
        for x, y_bb, y_class in train_dl: #x = [batch_size, RGB, 300, 300]
            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_bb = y_bb.to(device).float()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class.squeeze(1), reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb.squeeze(1), reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
            _, pred = torch.max(out_class, 1)
            pred = (pred==2).float()
            y_class = (y_class==2).float().squeeze(1)
            accuracy += pred.eq(y_class).sum().item()

        train_loss = sum_loss/total
        train_acc = accuracy/total
        # test_loss, test_acc = val_metrics(model, test_dl, C)
        print("for epoch: %i \t train_loss %.3f train_accuracy %.3f  " % (i, train_loss, train_acc))

        # print("for epoch: %f \t train_loss %.3f train_accuracy %.3f \t test_loss %.3f,test_accuracy %.3f  " % (i, train_loss, train_acc,test_loss, test_acc))

def main():
    # Learning parameters
    batch_size = 64  # batch size
    workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BB_model().to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.006)
    train_dataset = mask_dataset(dataset='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)
    test_dataset = mask_dataset(dataset='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=workers,
                                               pin_memory=True)

    train_epocs(model, device, optimizer, train_loader,test_loader, epochs=15)

if __name__ == '__main__':
    main()