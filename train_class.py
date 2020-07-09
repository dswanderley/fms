import copy
import utils
import time
import torch
import torch.nn as nn
import torchvision.models as models
from dataset import Dataset
from transforms import get_transform
import numpy as np


pretrained = True
num_epochs = 50
batch_size = 72
num_workers = 4
DATA_DIR = '/home/master/dataset/train/'

model_name = 'resnext50'

SAVE_MODEL = (model_name + '_classifier')
load_weigths = False

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_model(model, dataloaders,
                criterion, optimizer, num_epochs=25,
                is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


""" Training script """
if __name__ == '__main__':

    print(device)

    if load_weigths:
        model = torch.load(SAVE_MODEL)
    else:
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif model_name == 'resnext50':
            model = models.resnext50_32x4d(pretrained=True)
        set_parameter_requires_grad(model, True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

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


    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    dataloaders_dict = {'train': data_loader,
                        'val': data_loader_val}

    # Train and evaluate
    model, hist = train_model(model,
                                dataloaders_dict,
                                criterion,
                                optimizer,
                                num_epochs=num_epochs,
                                is_inception=(model_name=="inception"))

    torch.save(model, model_name + "_final")

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

