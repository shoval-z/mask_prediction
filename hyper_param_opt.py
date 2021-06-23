import wandb
from torch import nn
import torch
from darknet53_yoloV3 import Darknet53, train_and_eval_single_epoch
import torchvision.models as models
import torch.nn.functional as F
from dataset import mask_dataset
import numpy as np
from utils import save_model
from utils import xy_to_cxcy, calc_iou
import warnings


def main():
    run = wandb.init()
    batch_size = run.config.batch_size
    lr = run.config.learnig_rate
    acc_loss_weight = run.config.accuracy_loss_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet53().to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.015)

    train_dataset = mask_dataset(dataset='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)
    test_dataset = mask_dataset(dataset='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)
    for epoch in range(30):
        train_iou, train_acc, train_loss, test_iou, test_acc, test_loss = train_and_eval_single_epoch(model, device,
                                                                                                      optimizer,
                                                                                                      train_loader,
                                                                                                      train_dataset,
                                                                                                      test_loader,
                                                                                                      test_dataset,
                                                                                                      epoch=epoch)

        wandb.log({"Train Accuracy": train_acc, "Train IoU": train_iou, train_acc, "Train Loss": train_loss,
        "Test Accuracy": test_acc, "Test IoU": test_iou, "Test Loss": test_loss, "epoch": epoch})

    # your parameters go here
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'Val Accuracy', 'goal': 'maximize'},
        'parameters': {
            'batch_size': {'values': [8, 16, 32, 64]},
            'learnig_rate': {'values': [0.001,0.005,0.01,0.015,0.02,0.025]},
            'accuracy_loss_weight': {'values': [0.2, 0.5, 1, 1.5, 2, 3]}
        }}

    # create new sweep
    sweep_id = wandb.sweep(sweep_config, entity="shovalz", project="mask_prediction_darknet53")

    # run the agent to execute the code
    wandb.agent(sweep_id, function=main)
