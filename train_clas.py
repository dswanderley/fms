import utils
import torch
import torch.nn as nn
import torchvision.models as models
from dataset import Dataset
from transforms import get_transform
from resnet_class import resnet18_classifier
import numpy as np


pretrained = True
num_epochs = 50
batch_size = 20
num_workers = 1
DATA_DIR = '../dataset/train/'

SAVE_MODEL = ('resnet18_classifier')
load_weigths = True

@torch.no_grad()
def evaluate_model(model, data_loader_val, criterion, epoch):

    model.eval()

    loss_cat = []

    for idx, (images, labels) in enumerate(data_loader_val):
        a = 1
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(images)

        loss = criterion(outputs, labels)

        error = labels - outputs

        if idx == 0:
            error_cat = error
        elif idx == 1:
            error_cat = torch.cat([error_cat, error], 0)

        loss_cat.append(loss.item())

    err = 100 * torch.mean(torch.abs(error_cat), 0)

    loss = np.sum(np.abs(loss_cat), 0)

    print("EPOCH", epoch, ":", "Fish error", err[0].item(), "Background error", err[1].item(), "Loss", loss)

    return loss, err

""" Training script """
if __name__ == '__main__':

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    if load_weigths:
        model = torch.load(SAVE_MODEL)
    else:
        backbone = models.resnet18(pretrained=pretrained)
        #model = models.resnext50_32x4d(pretrained=pretrained)

        model = resnet18_classifier(backbone)

    #if load_weigths:
    #    model = torch.load(SAVE_MODEL)
    model.to(device)

    # use our dataset and defined transformations
    dataset = Dataset(DATA_DIR, transforms=get_transform(train=True))
    dataset_val = Dataset(DATA_DIR, transforms=get_transform(train=False))

    # split the dataset into train and validation sets
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    dataset_sub = torch.utils.data.Subset(dataset, indices[:-500])
    dataset_val_sub = torch.utils.data.Subset(dataset_val, indices[-500:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=utils.collate_fn_class
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn_class
    )


    criterion = nn.BCELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    best_loss = float("inf")

    for epoch in range(num_epochs):

        print("epoch", epoch)

        # TRAIN
        model.train()

        batch_num = 0
        for idx, (images, labels) in enumerate(data_loader):

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            print("batch", idx, "/", len(data_loader)-1, "loss :", loss.item())

        new_loss, error = evaluate_model(model, data_loader_val, criterion, epoch)

        if best_loss > new_loss:
            torch.save(model, "resnet18_classifier")
            print('Model Saved. error', error)
            best_loss = new_loss

    torch.save(model, "resnet18_final")

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    