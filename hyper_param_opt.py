import wandb
import torch
from darknet53_yoloV3 import Darknet53, train_and_eval_single_epoch
from dataset import mask_dataset
from utils import save_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # to ignore the .to(dtype=torch.uint8) warning message


def main():
    run = wandb.init()
    batch_size = run.config.batch_size
    lr = run.config.learnig_rate
    acc_loss_weight = run.config.accuracy_loss_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet53().to(device)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    train_dataset = mask_dataset(dataset='train', path=f'/home/student/train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)
    test_dataset = mask_dataset(dataset='test', path=f'/home/student/test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)
    for epoch in range(30):
        train_iou, train_acc, train_loss, test_iou, test_acc, test_loss, new_model = train_and_eval_single_epoch(model, device,
                                                                                                      optimizer,
                                                                                                      train_loader,
                                                                                                      train_dataset,
                                                                                                      test_loader,
                                                                                                      test_dataset,
                                                                                                      epoch,
                                                                                                      acc_loss_weight)

        wandb.log({"Train Accuracy": train_acc, "Train IoU": train_iou, "Train Loss": train_loss,
                   "Test Accuracy": test_acc, "Test IoU": test_iou, "Test Loss": test_loss, "epoch": epoch})
        # if test_iou > 0.55:
        #     print('saving model')
        #     save_model(f'{epoch}lr:{lr},bs:{batch_size},loss_weight:{acc_loss_weight}_darknet53', new_model)


# your parameters go here
sweep_config = {
    'method': 'random',
    'metric': {'name': 'Test IoU', 'goal': 'maximize'},
    'parameters': {
        'batch_size': {'values': [32, 16]},
        'learnig_rate': {'values': [0.001, 0.005, 0.01, 0.015, 0.02, 0.025]},
        'accuracy_loss_weight': {'values': [0.5, 1, 1.5, 2, 3]},

    }}

# create new sweep
sweep_id = wandb.sweep(sweep_config, entity="shovalz", project="mask_prediction")

# run the agent to execute the code
wandb.agent(sweep_id, function=main)
