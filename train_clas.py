import utils
import torch
import torch.nn as nn
import torchvision.models as models
from dataset import Dataset
from transforms import get_transform


pretrained = True
num_epochs = 30
batch_size = 4
num_workers = 1
DATA_DIR = '../dataset/train/' 

""" Training script """
if __name__ == '__main__':

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    model = models.resnet18(pretrained=pretrained)
    #model = models.resnext50_32x4d(pretrained=pretrained)
    model.fc = nn.Linear(512, 2)

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
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters())


    for epoch in range(num_epochs):

        # TRAIN
        model.train()
        for images, labels in data_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        # VAL
        

# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    