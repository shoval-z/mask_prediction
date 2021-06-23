from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from dataset import mask_dataset
import numpy as np
from utils import save_model
from utils import xy_to_cxcy,calc_iou
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # to ignore the .to(dtype=torch.uint8) warning message



class BB_model(nn.Module):
    def __init__(self):
        super(BB_model, self).__init__()
        resnet = models.resnet34(pretrained=False)
        # resnet = models.resnet18(pretrained=False)
        # resnet50 = models.resnet50(pretrained=False)
        # resnet152 = models.resnet152(pretrained=False)
        layers = list(resnet.children())[:8]
        # layers = list(resnet50.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.classifier = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 3))
        self.bb = nn.Sequential(nn.BatchNorm1d(512), nn.Linear(512, 4))
        # self.classifier = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, 3))
        # self.bb = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, 4))

    def forward(self, x):
        x = self.features1(x) # x =[batch_size, 128, 38, 38]  # x =[batch_size, 512, 38, 38]
        x = self.features2(x) #x = [batch_size, 512, 10, 10]  # x = [batch_size, 2048, 10, 10]
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x) # x=[batch_size, 512, 1, 1]
        x = x.view(x.shape[0], -1) #x = [batch_size, 512]
        classifier = self.classifier(x) #[batch_size, num_class]
        bb = self.bb(x) #[batch_size, 4]
        return classifier, bb

def val_metrics(model, device, valid_dl,test_dataset, C=1):
    model.eval()
    total = 0
    sum_loss = 0
    sum_iou = 0
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

        # y_bb = [b.to(device) for b in y_bb]
        # y_class = [l.to(device) for l in y_class]

        origin_size = test_dataset.image_sizes[idx]
        origin_size = origin_size.squeeze(1).to(device)

        out_bb = torch.mul(out_bb, origin_size)
        out_bb = xy_to_cxcy(out_bb)

        y_bb = torch.mul(y_bb.squeeze(1), origin_size)
        y_bb = xy_to_cxcy(y_bb)

        tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                   zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
        sum_iou += np.sum(tmp_iou)

        total += batch
    return sum_iou/total, correct/total, sum_loss/total


def train_epocs(model,device, optimizer, train_dl, train_dataset, epochs, C=1, init_epoch=0):
    print('start training')
    loss_list, iou_list, acc_list = list(), list(), list()
    for i in range(epochs):
        i += init_epoch
        model.train()
        total = 0
        sum_loss = 0
        sum_iou = 0
        accuracy = 0
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

            y_bb = torch.mul(y_bb.squeeze(1), origin_size)
            y_bb = xy_to_cxcy(y_bb)

            tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                       zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
            sum_iou += np.sum(tmp_iou)

        print('saving model')
        save_model(f'{i}_basic_model_resnet34', model)

        train_loss = sum_loss / total
        train_acc = accuracy / total
        train_iou = sum_iou / total
        print("for epoch: %f \t train_iou %.3f,train_accuracy %.3f,train_loss %.3f  " % (
        i, train_iou, train_acc, train_loss))
        loss_list.append(train_loss)
        iou_list.append(train_iou)
        acc_list.append(train_acc)
    return iou_list, acc_list, loss_list

def update_optimizer(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr

def main():
    # Learning parameters
    batch_size = 16  # batch size
    workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint = f'19_basic_model_checkpoint_ssd300.pth.tar'
    # checkpoint = torch.load(checkpoint)
    # model = BB_model()
    # model.load_state_dict(checkpoint['model'])
    # model = model.to(device)
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=0.006)
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # model = BB_model().to(device)
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=0.001)
    #
    # train_dataset = mask_dataset(dataset='train')
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                                            num_workers=workers,
    #                                            pin_memory=True)
    #
    # iou_list, acc_list, loss_list = train_epocs(model, device, optimizer, train_loader, train_dataset=train_dataset, epochs=40, C=1,init_epoch=0)
    # print('train_iou=', iou_list)
    # print('train_accuracy=', acc_list)
    # print('train_loss=', loss_list)
    # del train_loader, train_dataset

    test_dataset = mask_dataset(dataset='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=workers,
                                               pin_memory=True)

    loss_list, iou_list, acc_list = list(), list(), list()
    for i in range(20,30):
        # checkpoint = f'{i}__basic_model_new_mean_std_checkpoint_ssd300.pth.tar'
        checkpoint = f'{i}_basic_model_checkpoint_ssd300.pth.tar'
        # checkpoint = torch.load(checkpoint)
        model = torch.load(checkpoint)
        # model = BB_model()
        # model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        test_iou, test_acc, test_loss = val_metrics(model, device, valid_dl=test_loader, test_dataset=test_dataset,
                                                    C=1)
        print("for epoch: %f \t test_iou %.3f,test_accuracy %.3f,test_loss %.3f  " % (i, test_iou, test_acc, test_loss))
        loss_list.append(test_loss)
        iou_list.append(test_iou)
        acc_list.append(test_acc)
        del model
    print('test_iou=', iou_list)
    print('test_accuracy=', acc_list)
    print('test_loss=', loss_list)

if __name__ == '__main__':
    # model = BB_model()
    # print(sum(p.numel() for p in model.parameters()))
    main()
