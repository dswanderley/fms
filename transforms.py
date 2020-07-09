import random
import torch
import kornia
import numpy as np
from torchvision.transforms import functional as F
import torchvision
from torchvision.transforms import transforms as T
import cv2

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

class RandomRotation(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            h, w = image.shape[1:]

            # Debug
            # from matplotlib import pyplot as plt
            # plt.figure()
            # plt.imshow(image.numpy().transpose(1, 2, 0))
            # plt.show()
            
            # rotate image in random[0, 40] degrees (right or left) 
            # from the center of the image
            angle = random.randint(0,40)
            print("rotation (degrees): ", angle)
            rotate = torchvision.transforms.RandomRotation(angle)
            image = rotate(F.to_pil_image(image))

            if len(target["boxes"]) > 0:
                # rotate bboxes with the detections
                corners = self.get_corners(target["boxes"])
                corners = np.hstack((corners, target["boxes"][:,4:]))
                corners[:,:8] = self.rotate_box(corners, angle, w//2, h//2, w, h)
                new_bboxes = self.get_enclosing_box(corners)

                # # Debug
                # import matplotlib.pyplot as plt
                # import matplotlib.patches as pat
                
                # fig, ax = plt.subplots(1)
                # plt.imshow(image)
                # for bbox in target["boxes"]:
                #     if bbox.shape[0] != 0:
                #         rect = pat.Rectangle([int(bbox[0]), int(bbox[1])], # x, y
                #                             int(bbox[2] - bbox[0]),  # w
                #                             int(bbox[3] - bbox[1]),  # h
                #                             edgecolor='g', linewidth=3, fill=False)
                #         ax.add_patch(rect)
                # for bbox in new_bboxes:
                #     if bbox.shape[0] != 0:
                #         rect = pat.Rectangle([int(bbox[0]), int(bbox[1])], # x, y
                #                             int(bbox[2] - bbox[0]),  # w
                #                             int(bbox[3] - bbox[1]),  # h
                #                             edgecolor='r', linewidth=3, fill=False)
                #         ax.add_patch(rect)
                # plt.show()

                target["boxes"] = new_bboxes

            image = F.to_tensor(image)

        return image, target

    def get_corners(self, bboxes):
        width = bboxes[:,[2]] - bboxes[:,[0]]
        height = bboxes[:,[3]] - bboxes[:,[1]]

        x1 = bboxes[:,[0]]
        y1 = bboxes[:,[1]]

        x2 = x1 + width
        y2 = y1

        x3 = x1
        y3 = y1 + height

        x4 = bboxes[:,[2]]
        y4 = bboxes[:,[3]]

        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

        return corners
        return -1

    def rotate_box(self, corners, angle, cx, cy, w, h):
        corners = corners.reshape(-1,2)
        corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M,corners.T).T
        
        calculated = calculated.reshape(-1,8)
        
        return calculated

    def get_enclosing_box(self, corners):
        x_ = corners[:,[0,2,4,6]]
        y_ = corners[:,[1,3,5,7]]
        
        xmin = np.min(x_,1).reshape(-1,1)
        ymin = np.min(y_,1).reshape(-1,1)
        xmax = np.max(x_,1).reshape(-1,1)
        ymax = np.max(y_,1).reshape(-1,1)
        
        final = np.hstack((xmin, ymin, xmax, ymax, corners[:,8:]))
        
        return final

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

class RandomScale(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            h, w = image.shape[1], image.shape[2]

            # Debug
            from matplotlib import pyplot as plt
            plt.figure()
            plt.imshow(image.numpy().transpose(1, 2, 0))
            plt.show()

            # transform to PIL image
            image = F.to_pil_image(image)
            # scale image
            # scale = T.RandomResizedCrop( (h, w), scale=(0.6, 1.0), ratio=(w/h, w/h), interpolation=2)
            # image = scale(image)

            # scale image
            s = random.uniform(0.5, 1.0)
            resize_scale = 1 + s
            #image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)
            scale = T.RandomResizedCrop( (h, w), scale=(s, s), ratio=(w/h, w/h), interpolation=2)
            image = scale(image)
            image = F.to_tensor(image)

            # scale bboxes
            if len(target["boxes"]) > 0:
                

                # Debug
                import matplotlib.pyplot as plt
                import matplotlib.patches as pat
                
                fig, ax = plt.subplots(1)
                plt.imshow(image.numpy().transpose(1, 2, 0))
                for bbox in target["boxes"]:
                    if bbox.shape[0] != 0:
                        rect = pat.Rectangle([int(bbox[0]), int(bbox[1])], # x, y
                                            int(bbox[2] - bbox[0]),  # w
                                            int(bbox[3] - bbox[1]),  # h
                                            edgecolor='g', linewidth=3, fill=False)
                        ax.add_patch(rect)

                bboxes = target["boxes"]
                bboxes[:,:4] *= resize_scale

                for bbox in bboxes:
                    if bbox.shape[0] != 0:
                        rect = pat.Rectangle([int(bbox[0]), int(bbox[1])], # x, y
                                            int(bbox[2] - bbox[0]),  # w
                                            int(bbox[3] - bbox[1]),  # h
                                            edgecolor='r', linewidth=3, fill=False)
                        ax.add_patch(rect)
                plt.show()
            
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
        transforms.append(RandomBlur(0.2))
        transforms.append(RandomContrastLuminance(0.2))
        #transforms.append(RandomRotation(1))
        #transforms.append(RandomScale(1))
        
    return Compose(transforms)

# transform for images only (no labels)
def get_test_transform():
    # in case you want to insert some transformation in here
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])