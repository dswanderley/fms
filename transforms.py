import random
import torch
import kornia
import numpy as np
from torchvision.transforms import functional as F
import torchvision
from torchvision.transforms import transforms as T

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomBlur(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):

        if random.random() < self.prob:
            # add B dimension to image: (C, H, W) to (B, C, H, W)
            blur_image = torch.unsqueeze(image.float(), dim=0) 
            # blur the image
            blur = kornia.filters.GaussianBlur2d((11, 11), (10.5, 10.5))
            blur_image = blur(blur_image)

            # Debug
            '''
            from matplotlib import pyplot as plt
            plt.figure()
            plt.imshow(image.numpy().transpose(1, 2, 0))
            plt.show()
            plt.figure()
            plt.imshow(blur_image[0].numpy().transpose(1, 2, 0))
            plt.show()
            '''
            image = blur_image[0]

        return image, target

class RandomContrastLuminance(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:

             # Debug
            # from matplotlib import pyplot as plt
            # plt.figure()
            # plt.imshow(image.numpy().transpose(1, 2, 0))
            # plt.show()

            # transform to PIL image
            if (image.shape==3):
                image = F.to_pil_image(image)
                # apply brightness to the image
                contrast = T.ColorJitter(brightness=1.0, contrast=0.5, saturation=0.0, hue=0.0)
                image = contrast(image)
                image = F.to_tensor(image)

                # Debug
                # plt.figure()
                # plt.imshow(image.numpy().transpose(1, 2, 0))
                # plt.show()
            
        return image, target

class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(1)   # flip img vertically
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]  # flip bbox upside down
            target["boxes"] = bbox

            # Debug
            """
            import matplotlib.pyplot as plt
            import matplotlib.patches as pat
            
            if bbox.shape[0] != 0:
                img = image
                fig, ax = plt.subplots(1)
                plt.imshow(  img.permute(1, 2, 0)  )
                rect = pat.Rectangle([int(bbox[0, 0]), int(bbox[0, 1])], # x, y
                                     int(bbox[0, 2] - bbox[0, 0]),  # w
                                     int(bbox[0, 3] - bbox[0, 1]),  # h
                                     edgecolor='r', linewidth=3, fill=False)
                ax.add_patch(rect)
                plt.show()
                a = 1
            """
            if "masks" in target:
                target["masks"] = target["masks"].flip(1)  # flip masks upside down

            # TODO: check this code
            """
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, height)
                target["keypoints"] = keypoints
            """
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

# Data augmentation
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
        transforms.append(RandomVerticalFlip(0.5))
        transforms.append(RandomBlur(0.15))
        transforms.append(RandomContrastLuminance(0.15))
        #transforms.append(RandomRotation(1))
        #transforms.append(RandomScale(1))
        
    return Compose(transforms)

# transform for images only (no labels)
def get_test_transform():
    # in case you want to insert some transformation in here
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])