import torch
import torchvision
from transforms import get_test_transform
from dataset import Test_Dataset
import csv
from torchvision.ops import nms
import numpy as np

NMS_THRESHOLD = 0.1
SAVED_MODEL = 'fasterRCNN'
DATA_DIR = '/home/master/dataset/test/'

# Load dataset
dataset_test = Test_Dataset(DATA_DIR, transforms=get_test_transform())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load model
model = torch.load(SAVED_MODEL)
model.to(device)

predictions = list()



def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].long()

    return keep



for ii, (img, seq, frame) in enumerate(dataset_test):
    if ii%50 == 0:
        print("Processed %d / %d images" % (ii, len(dataset_test)))

    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    boxes = prediction[0]['boxes'].cpu()
    scores = prediction[0]['scores'].cpu()

    nms_indices = soft_nms_pytorch(boxes, scores, NMS_THRESHOLD)

    nms_boxes = boxes[nms_indices].tolist()
    nms_scores = scores[nms_indices].tolist()
        
    # if there are no detections there is no need to include that entry in the predictions
    if len(nms_boxes) > 0:
        for bb, score in zip(nms_boxes, nms_scores):
            predictions.append([seq, frame, list(bb), score])


with open('predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(['seq', 'frame', 'label', 'score'])
    writer.writerows(predictions)
