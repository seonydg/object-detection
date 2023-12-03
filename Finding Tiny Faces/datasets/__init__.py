#init file chanaged




import numpy as np
import os
import os.path as osp
import json
from utils.cluster import compute_kmedoids
from .wider_face import WIDERFace
from torch.utils import data


def get_dataloader(datapath, num_templates=25,
                   template_file="templates.json", img_transforms=None,
                   train=True, split="train"):
    template_file = os.path.join("datasets", template_file)
    traindata='data/WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
    dataset_root='data/WIDER'
    batch_size=1
    workers=8

    if os.path.exists(template_file):
        templates = json.load(open(template_file))

    # else:
    #     # Cluster the bounding boxes to get the templates
    #     dataset = WIDERFace(osp.expanduser(traindata), [])
    #     clustering = compute_kmedoids(dataset.get_all_bboxes(), 1, indices=num_templates,
    #                                   option='pyclustering', max_clusters=num_templates)

    #     print("Canonical bounding boxes computed")
    #     templates = clustering[num_templates]['medoids'].tolist()

    #     # record templates
    #     json.dump(templates, open(template_file, "w"))

    templates = np.round_(np.array(templates), decimals=8)

    data_loader = data.DataLoader(WIDERFace(datapath, templates,
                                            train=train, split=split, img_transforms=img_transforms,
                                            dataset_root=dataset_root,
                                            debug=False),
                                  batch_size=batch_size, shuffle=train,
                                  num_workers=workers, pin_memory=True)

    return data_loader, templates
