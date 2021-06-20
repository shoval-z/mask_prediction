from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from dataset import mask_dataset
from utils import save_model
from utils import calc_iou, xy_to_cxcy
import numpy as np


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

def val_metrics(model, device, valid_dl,test_dataset ,C=1000):
    model.eval()
    total = 0
    sum_iou = 0
    sum_loss = 0
    correct = 0
    for idx,(x, y_bb, y_class) in enumerate(valid_dl):
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
        y_bb = [b.to(device) for b in y_bb]
        # y_class = [l.to(device) for l in y_class]

        origin_size = test_dataset.image_sizes[idx]
        origin_size = origin_size.squeeze(1).to(device)

        out_bb = torch.mul(out_bb, origin_size)
        out_bb = xy_to_cxcy(out_bb)

        y_bb = torch.mul(torch.stack(y_bb).squeeze(1), origin_size)
        y_bb = xy_to_cxcy(y_bb)

        tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
        sum_iou += np.sum(tmp_iou)
        total += batch
    return sum_iou/total, correct/total, sum_loss/total


def train_epocs(model,device, optimizer, train_dl, train_dataset, epochs=10,C=1000):
    idx = 0
    print('start train')
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        sum_iou = 0
        accuracy =0
        for idx, (x, y_bb, y_class) in enumerate(train_dl): #x = [batch_size, RGB, 300, 300]
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
            # idx += 1
            total += batch
            sum_loss += loss.item()
            _, pred = torch.max(out_class, 1)
            pred = (pred==2).float()
            y_class = (y_class==2).float().squeeze(1)
            accuracy += pred.eq(y_class).sum().item()

            origin_size = train_dataset.image_sizes[idx]
            origin_size = origin_size.squeeze(1).to(device)

            out_bb = torch.mul(out_bb, origin_size)
            out_bb = xy_to_cxcy(out_bb)

            y_bb = torch.mul(torch.stack(y_bb).squeeze(1), origin_size)
            y_bb = xy_to_cxcy(y_bb)

            tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                       zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
            sum_iou += np.sum(tmp_iou)

        print('saving model')
        save_model(f'{i}_basic_model', model)

        train_loss = sum_loss/total
        train_acc = accuracy/total
        train_iou = sum_iou/total
        print("for epoch: %f \t test_iou %.3f,test_accuracy %.3f,test_loss %.3f  " % (i, train_iou, train_acc,train_loss))


def main():
    # Learning parameters
    batch_size = 16  # batch size
    workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BB_model().to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.006)
    train_dataset = mask_dataset(dataset='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)
    train_epocs(model, device, optimizer, train_loader,train_dataset=train_dataset, epochs=15)
    del train_loader,train_dataset


    test_dataset = mask_dataset(dataset='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=workers,
                                               pin_memory=True)

    for i in range(15):
        checkpoint = f'{i}_basic_model_checkpoint_ssd300.pth.tar'
        checkpoint = torch.load(checkpoint)
        model = BB_model()
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        test_iou, test_acc, test_loss = val_metrics(model, device, valid_dl=test_loader,test_dataset=test_dataset,C=1000)
        print("for epoch: %f \t test_iou %.3f,test_accuracy %.3f,test_loss %.3f  " % (i, test_iou, test_acc,test_loss))
        del model

if __name__ == '__main__':
    main()
