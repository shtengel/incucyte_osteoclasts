import numpy as np
from skimage.morphology import dilation, disk

def count_cell_neighbors(segmentation_mask, buffer=2):
    labels = np.unique(segmentation_mask)
    labels = labels[labels != 0]

    adjacency = {l: set() for l in labels}
    touch_mask = np.zeros_like(segmentation_mask, dtype=bool)

    for i, l1 in enumerate(labels):
        mask1 = segmentation_mask == l1
        dil1 = dilation(mask1) # , disk(buffer))

        for l2 in labels[i+1:]:
            mask2 = segmentation_mask == l2
            dil2 = dilation(mask2) # , disk(buffer))

            overlap = (dil1 & mask2) | (dil2 & mask1)
            if overlap.any():
                adjacency[l1].add(l2)
                adjacency[l2].add(l1)
                touch_mask |= overlap

    neighbor_counts = sum([len(neigh) for cell, neigh in adjacency.items()]) / 2
    return adjacency, neighbor_counts, touch_mask
