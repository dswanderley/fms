import os
import argparse
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from engine import train_one_epoch, evaluate
import utils
from dataset import Dataset
from transforms import get_transform
from models.backbones import get_backbone, get_model
from models.fpn import GroupedPyramidFeatures
from models.deeplab import DeepLabv3Plus


""" Training parameters """
# Save condition
val_mAP = 0


""" Training script """
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument("--backbone", type=str, default="resnet18", help="backbone name")
    parser.add_argument("--neck", type=str, default="deeplab", help="network neck name")
    parser.add_argument("--num_epochs", type=int, default=50, help="size of each image batch")
    parser.add_argument("--batch_size", type=int, default=4, help="number of workers")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--data_dir", type=str, default="local", help="dataset dir")
    parser.add_argument("--load_weights", type=int, default=0, help="to load weights")
    parser.add_argument("--min_size", type=int, default=600, help="image minimum size")
    parser.add_argument("--max_size", type=int, default=600, help="maximum size")

    opt = parser.parse_args()
    print(opt)

    # Dataset path
    if opt.data_dir == 'vm':
        DATA_DIR = '/home/master/dataset/train/'   # VISUM VM path
    else:
        DATA_DIR = '../dataset/train/'              # Your PC path, don't forget the backslash in the end

    # Get sequence list
    sequences = [ int(folder.replace('seq','')) for _, dirnames, filenames in os.walk(DATA_DIR) for folder in dirnames if 'seq' in folder ]
    val_len = int(len(sequences) / 5)
    seq_train = sequences[:-val_len]
    seq_val = sequences[-val_len:]

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Backbone name
    backbone_name = opt.backbone
    # Neck name
    neck_name  = opt.neck
    # Weights definitions
    load_weigths = True if opt.load_weights == 1 else False
    SAVE_MODEL = ('fasterRCNN_' + str(backbone_name) + '_' + str(neck_name) )

    # number of processes 
    num_workers = opt.num_workers         # 4 for VISUM VM and 1 for our Windows machines

    # Training epochs
    num_epochs = opt.num_epochs

    # Number of images in a batch
    batch_size = opt.batch_size

    # Image size
    max_size = opt.max_size
    min_size = opt.min_size
    
    # Initial model 
    model = fasterrcnn_resnet50_fpn(pretrained=True, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    # load a pre-trained model for classification and return
    # only the features
    if neck_name == 'fpn':
        out_channels = 256
        backbone =  resnet_fpn_backbone(backbone_name, pretrained=True)
        backbone.out_channels = out_channels
        model.backbone = backbone
    elif neck_name == 'gfpn':
        out_channels = 256
        backbone = GroupedPyramidFeatures(backbone_name=backbone_name, out_features=out_channels, pretrained=True)       
        backbone.out_channels = out_channels 
        model.backbone = backbone
    elif neck_name == 'deeplab':
        out_channels = 256
        backbone = DeepLabv3Plus(n_classes=out_channels, backbone_name=backbone_name, pretrained=True)
        backbone.out_channels = out_channels
        model.backbone = backbone
    else:
        backbone, out_channels = get_backbone(backbone_name, pretrained=True)
        backbone.out_channels = out_channels

        # Anchors
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"], output_size=7, sampling_ratio=2
        )

        # put the pieces together inside a FasterRCNN model
        # one class for fish, other for the backgroud
        model = FasterRCNN(
            backbone,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=min_size, max_size=max_size
        )

    # See the model architecture
    print(model)

    if load_weigths:
        model = torch.load(SAVE_MODEL)
    model.to(device)

    # define an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # use our dataset and defined transformations
    dataset_train = Dataset(DATA_DIR, transforms=get_transform(train=True), sequences=seq_train)
    dataset_val = Dataset(DATA_DIR, transforms=get_transform(train=False), sequences=seq_val)
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=utils.collate_fn
    )

    # Training
    for epoch in range(int(num_epochs)):
        # train for one epoch, printing every 10 iterations
        epoch_loss = train_one_epoch(model, optimizer, data_loader,
                                        device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the validation dataset
        evaluator = evaluate(model, data_loader_val, dataset_val, device)

        if val_mAP < evaluator[0]:
            val_mAP = evaluator[0]
            torch.save(model, SAVE_MODEL)
            print('Model Saved. mAP = %1.6f' % val_mAP)
