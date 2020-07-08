import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from utils.datasets import OvaryDataset
from torch.utils.data import DataLoader
import utils
import torch
from dataset import Dataset
from transforms import get_transform

DATA_DIR = '../dataset/train/'  # Your PC path, don't forget the backslash in the end

class Clusters():
    def __init__(self):
        self.N_CLUSTERS = 6    # CHANGE THIS WHEN THIS CHANGE!!!
        self.batch_size = 12   # CHANGE THIS WHEN THIS CHANGE!!!
        self.num_workers = 1   # CHANGE THIS WHEN THIS CHANGE!!!

        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.widths = []
        self.heights = []

if __name__ == '__main__':

    # init cluster vars
    clusters = Clusters()

    # use our dataset and defined transformations
    dataset = Dataset(DATA_DIR, transforms=get_transform(train=True))

    # create data loader
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=clusters.batch_size, shuffle=True, num_workers=clusters.num_workers, collate_fn=utils.collate_fn
        )

    # iterate through the labels
    for images, targets, _ in data_loader:
        # read each image in batch
        for target in targets:
            for bbox in target['boxes']:
                w = int(bbox[2].item() - bbox[0].item())  # width
                h = int(bbox[3].item() - bbox[1].item())  # hight
                clusters.widths.append(w)
                clusters.heights.append(h)

    # compute clusters
    data = np.array( [clusters.widths, clusters.heights] )
    kmeans = KMeans(n_clusters=clusters.N_CLUSTERS, random_state=0).fit(data.transpose())

    # print width vs hight graph: (x, y) of each detection
    for xy, cl in zip(data.transpose(), kmeans.labels_):
        x = xy[0]
        y = xy[1]
        c = clusters.colors[cl]
        plt.scatter(x, y, c=c, marker=".")

    # print width vs hight graph: clusters 
    centers = [ [c[0], c[1], c[0]*c[1]] for c in kmeans.cluster_centers_]
    centers = np.array(centers)
    centers = np.sort(centers.view('i8,i8,i8'), order=['f1'], axis=0).view(np.float)

    for w, h, area in centers:
        print('width: ' + str( round(w) ) , 'height: ' + str( round(h) ), 'area: ' + str( round(area) ))
        plt.scatter(w, h, c='k', marker="*")

    # show width vs hight graph:
    plt.axis([ 0, max(clusters.widths)+5, 0, max(clusters.heights)+5 ])
    plt.show()

    aspect_ratios =  []
    # print aspect ratio graph: (width/hight) of each detection
    for bbox in data.transpose():
        aspect_ratios.append(bbox[0]/bbox[1])

    # show graph
    plt.hist(aspect_ratios, bins=100)
    min, max = plt.ylim()
    plt.vlines(np.quantile(aspect_ratios, 0.25),ymin=min, ymax=max)
    plt.vlines(np.quantile(aspect_ratios, 0.5),ymin=min, ymax=max)
    plt.vlines(np.quantile(aspect_ratios, 0.75),ymin=min, ymax=max)
    plt.show()