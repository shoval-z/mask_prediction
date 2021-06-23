from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F
from dataset import mask_dataset
import numpy as np
from utils import save_model
from utils import xy_to_cxcy,calc_iou
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)  # to ignore the .to(dtype=torch.uint8) warning message


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.darknet53 = self.darknet_53()
        self.bbox = nn.Sequential(
            nn.Conv2d(1024, 5, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(5),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid()
        )

    def darknet_53(self):
        return nn.Sequential(
            convolutional_layer(3, 32, kernel_size=3, padding=1, stride=1),
            convolutional_layer(32, 64, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=64, out_channels=32) for _ in range(1)]),  # residual
            convolutional_layer(64, 128, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=128, out_channels=64) for _ in range(2)]),  # residual
            convolutional_layer(128, 256, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=256, out_channels=128) for _ in range(8)]),  # residual
            convolutional_layer(256, 512, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=512, out_channels=256) for _ in range(8)]),  # residual
            convolutional_layer(512, 1024, kernel_size=3, padding=1, stride=2),
            nn.Sequential(*[residual_block(in_channels=1024, out_channels=512) for _ in range(4)]),  # residual
        )

    def forward(self, image: torch.Tensor):
        features = self.darknet53(image) # [batch size, 1024, 10, 10]
        out = self.bbox(features).flatten(start_dim=1)
        classifier = out[:, 0]
        bbox = out[:, 1:]
        return classifier, bbox


def convolutional_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(residual_block, self).__init__()
        self.conv = nn.Sequential(
            convolutional_layer(in_channels, out_channels, kernel_size=1, padding=0, stride=1),
            convolutional_layer(out_channels, in_channels, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        return x + self.conv(x)


def train_epocs(model,device, optimizer, train_dl, train_dataset, epochs, C=1, init_epoch=0):
    print('start training')
    model.train()
    loss_list, iou_list, acc_list = list(), list(), list()
    loss_bb = torch.nn.L1Loss()
    loss_class = torch.nn.BCELoss()
    for i in range(epochs):
        i += init_epoch
        model.train()
        total = 0
        sum_loss = 0
        sum_iou = 0
        accuracy = 0
        for idx, (x, y_bb, y_class) in enumerate(train_dl): #x = [batch_size, RGB, 300, 300]

            origin_size = train_dataset.image_sizes[idx]
            origin_size = origin_size.squeeze(1).to(device)

            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_class = (y_class == 2).squeeze(1).float()
            y_bb = y_bb.to(device).float()
            out_class, out_bb = model(x)

            out_bb = torch.mul(out_bb, origin_size)
            out_bb = xy_to_cxcy(out_bb)

            y_bb = torch.mul(y_bb.squeeze(1), origin_size)
            y_bb = xy_to_cxcy(y_bb)

            # loss_class = F.cross_entropy(out_class, y_class.squeeze(1), reduction="sum")
            loss = loss_bb(out_bb, y_bb.squeeze(1))
            loss += 2*(loss_class(out_class, y_class))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += batch
            sum_loss += loss.item()
            # _, pred = torch.max(out_class, 1)
            pred = (out_class > 0.5).float()
            # y_class = (y_class == 2).float().squeeze(1)
            accuracy += pred.eq(y_class).sum().item()

            tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                       zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
            sum_iou += np.sum(tmp_iou)

        print('saving model')
        save_model(f'{i}_darknet53', model)

        train_loss = sum_loss / total
        train_acc = accuracy / total
        train_iou = sum_iou / total
        print("for epoch: %f \t train_iou %.3f,train_accuracy %.3f,train_loss %.3f  " % (
            i, train_iou, train_acc, train_loss))
        loss_list.append(train_loss)
        iou_list.append(train_iou)
        acc_list.append(train_acc)
    return iou_list, acc_list, loss_list


def val_metrics(model, device, valid_dl,test_dataset, C=1):
    print('strat eval')
    pred_lst, real_lst = [],[]
    model.eval()
    total = 0
    sum_loss = 0
    sum_iou = 0
    correct = 0
    loss_bb = torch.nn.L1Loss()
    loss_class = torch.nn.BCELoss()
    with torch.no_grad():
        for idx,(x, y_bb, y_class) in enumerate(valid_dl):
            origin_size = test_dataset.image_sizes[idx]
            origin_size = origin_size.squeeze(1).to(device)

            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_class = (y_class == 2).squeeze(1).float()
            y_bb = y_bb.to(device).float()
            out_class, out_bb = model(x)

            out_bb = torch.mul(out_bb, origin_size)
            out_bb = xy_to_cxcy(out_bb)

            y_bb = torch.mul(y_bb.squeeze(1), origin_size)
            y_bb = xy_to_cxcy(y_bb)

            loss = loss_bb(out_bb, y_bb.squeeze(1))
            loss += 2*(loss_class(out_class, y_class))
            # loss_bb = loss_bb.sum()
            # loss = loss_class + loss_bb/C
            # _, pred = torch.max(out_class, 1)
            pred = (out_class > 0.5).float()
            # pred = (pred == 2).float()
            # y_class = (y_class == 2).float().squeeze(1)
            correct += pred.eq(y_class).sum().item()
            sum_loss += loss.item()

            # y_bb = [b.to(device) for b in y_bb]
            # y_class = [l.to(device) for l in y_class]
            real_lst.extend(y_class)
            pred_lst.extend(pred)
            tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                       zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
            sum_iou += np.sum(tmp_iou)

            total += batch
    return sum_iou/total, correct/total, sum_loss/total, real_lst, pred_lst

def train_and_eval(model, device, optimizer, train_loader, train_dataset,test_loader, test_dataset, epochs=50,):
    print('start training')
    model.train()
    train_loss_list, train_iou_list, train_acc_list = list(), list(), list()
    loss_bb = torch.nn.L1Loss()
    loss_class = torch.nn.BCELoss()
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        sum_iou = 0
        accuracy = 0
        for idx, (x, y_bb, y_class) in enumerate(train_loader): #x = [batch_size, RGB, 300, 300]

            origin_size = train_dataset.image_sizes[idx]
            origin_size = origin_size.squeeze(1).to(device)

            batch = y_class.shape[0]
            x = x.to(device).float()
            y_class = y_class.to(device)
            y_class = (y_class == 2).squeeze(1).float()
            y_bb = y_bb.to(device).float()
            out_class, out_bb = model(x)

            out_bb = torch.mul(out_bb, origin_size)
            out_bb = xy_to_cxcy(out_bb)

            y_bb = torch.mul(y_bb.squeeze(1), origin_size)
            y_bb = xy_to_cxcy(y_bb)

            # loss_class = F.cross_entropy(out_class, y_class.squeeze(1), reduction="sum")
            loss = loss_bb(out_bb, y_bb.squeeze(1))
            loss += 2*(loss_class(out_class, y_class))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += batch
            sum_loss += loss.item()
            # _, pred = torch.max(out_class, 1)
            pred = (out_class > 0.5).float()
            # y_class = (y_class == 2).float().squeeze(1)
            accuracy += pred.eq(y_class).sum().item()

            tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                       zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
            sum_iou += np.sum(tmp_iou)

        print('saving model')
        save_model(f'{i}_darknet53', model)

        train_loss = sum_loss / total
        train_acc = accuracy / total
        train_iou = sum_iou / total
        print("for epoch: %f \t train_iou %.3f,train_accuracy %.3f,train_loss %.3f  " % (
            i, train_iou, train_acc, train_loss))
        train_loss_list.append(train_loss)
        train_iou_list.append(train_iou)
        train_acc_list.append(train_acc)

        ## eval
        print('strat eval')
        model.eval()
        total = 0
        sum_loss = 0
        sum_iou = 0
        correct = 0
        test_loss_list, test_iou_list, test_acc_list = list(), list(), list()
        loss_bb = torch.nn.L1Loss()
        loss_class = torch.nn.BCELoss()
        with torch.no_grad():
            for idx, (x, y_bb, y_class) in enumerate(test_loader):
                origin_size = train_dataset.image_sizes[idx]
                origin_size = origin_size.squeeze(1).to(device)

                batch = y_class.shape[0]
                x = x.to(device).float()
                y_class = y_class.to(device)
                y_class = (y_class == 2).squeeze(1).float()
                y_bb = y_bb.to(device).float()
                out_class, out_bb = model(x)

                out_bb = torch.mul(out_bb, origin_size)
                out_bb = xy_to_cxcy(out_bb)

                y_bb = torch.mul(y_bb.squeeze(1), origin_size)
                y_bb = xy_to_cxcy(y_bb)

                # loss_class = F.cross_entropy(out_class, y_class.squeeze(1), reduction="sum")
                loss = loss_bb(out_bb, y_bb.squeeze(1))
                loss += 2 * (loss_class(out_class, y_class))
                total += batch
                sum_loss += loss.item()
                # _, pred = torch.max(out_class, 1)
                pred = (out_class > 0.5).float()
                # y_class = (y_class == 2).float().squeeze(1)
                accuracy += pred.eq(y_class).sum().item()

                tmp_iou = [calc_iou(det_b, true_b) for det_b, true_b in
                           zip(out_bb.cpu().detach().numpy(), y_bb.squeeze(1).cpu().detach().numpy())]
                sum_iou += np.sum(tmp_iou)

            test_loss = sum_loss / total
            test_acc = accuracy / total
            test_iou = sum_iou / total
            print("for epoch: %f \t test_iou %.3f,test_accuracy %.3f,test_loss %.3f  " % (
                i, test_iou, test_acc, test_loss))
            test_loss_list.append(test_loss)
            test_iou_list.append(test_iou)
            test_acc_list.append(test_acc)

        print('train_iou=', train_iou_list)
        print('train_acc=', train_acc_list)
        print('train_loss=', train_loss_list)
        print('test_iou=', test_iou_list)
        print('test_acc=', test_acc_list)
        print('test_loss=', test_loss_list)


def main():
    # Learning parameters
    batch_size = 32  # batch size
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

    model = Darknet53().to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    #
    # train_dataset = mask_dataset(dataset='train')
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                                            num_workers=workers,
    #                                            pin_memory=True)


    # iou_list, acc_list, loss_list = train_epocs(model, device, optimizer, train_loader, train_dataset=train_dataset,
    #                                             epochs=50, C=1, init_epoch=0)
    # print('train_iou=', iou_list)
    # print('train_accuracy=', acc_list)
    # print('train_loss=', loss_list)
    # del train_loader, train_dataset

    test_dataset = mask_dataset(dataset='test', path='example_images')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=workers,
                                              pin_memory=True)

    # train_and_eval(model, device, optimizer, train_loader, train_dataset,test_loader,test_dataset, epochs=50)

    # loss_list, iou_list, acc_list = list(), list(), list()
    # for i in range(50):
    #     # checkpoint = f'{i}__basic_model_new_mean_std_checkpoint_ssd300.pth.tar'
    #     checkpoint = f'{i}_darknet53_checkpoint_ssd300.pth.tar'
    #     # checkpoint = torch.load(checkpoint)
    #     model = torch.load(checkpoint)
    #     # model = BB_model()
    #     # model.load_state_dict(checkpoint['state_dict'])
    #     model = model.to(device)
    #     test_iou, test_acc, test_loss = val_metrics(model, device, valid_dl=test_loader, test_dataset=test_dataset,
    #                                                 C=1)
    #     print("for epoch: %f \t test_iou %.3f,test_accuracy %.3f,test_loss %.3f  " % (i, test_iou, test_acc, test_loss))
    #     loss_list.append(test_loss)
    #     iou_list.append(test_iou)
    #     acc_list.append(test_acc)
    #     del model
    # print('test_iou=', iou_list)
    # print('test_accuracy=', acc_list)
    # print('test_loss=', loss_list)

    _, _,_, real_lst, pred_lst = val_metrics(model, device, test_loader, test_dataset, C=1)
    c_m = confusion_matrix(real_lst, pred_lst, labels=["Proper mask", "Not Proper mask"])
    sns.heatmap(c_m, linewidths=.5,cmap="YlGnBu")
    plt.show()


if __name__ == '__main__':
    main()

