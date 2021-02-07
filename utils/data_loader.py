
import os

import numpy as np

from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import build_dataloader


def pairing_input_target(inputs, targets) :
    '''

    inputs :  training or validation data
    target :  mask

    :return only returns the inputs that can be paired with target
    '''

    postfix_input = inputs[0].split(".")[-1];
    postfix_target = targets[0].split(".")[-1];

    set_in = set(map(lambda x: x.split(".")[0], inputs));
    set_tar = set(map(lambda x: x.split(".")[0], targets));

    pairs = set_tar.intersection(set_in);

    return sorted(set(map(lambda x: x + "." + postfix_input, pairs))),\
           sorted(set(map(lambda x: x + "." + postfix_target, pairs)));


def MMCVDataLoader(train_pipeline, val_pipeline, opt) :

    data_root = os.path.join("datasets/", opt.exp_name);

    mask_dir = os.path.join(data_root, "SegmentationClass");
    train_dir = os.path.join(data_root, "train");
    val_dir = os.path.join(data_root, "val");

    mask_imgs = os.listdir(mask_dir);
    train_imgs = os.listdir(train_dir);
    val_imgs = os.listdir(val_dir);

    train_imgs, target_train_imgs = pairing_input_target(train_imgs, mask_imgs);
    val_imgs, target_val_imgs = pairing_input_target(val_imgs, mask_imgs);

    train_split = os.path.join(data_root, "train.txt");
    val_split = os.path.join(data_root, "val.txt");

    with open(train_split,'w') as train_f :
        for img in train_imgs :
            train_f.write(img.split(".")[0]+"\n");

    with open(val_split,'w') as val_f :
        for img in val_imgs :
            val_f.write(img.split(".")[0]+"\n");


    trainDataSet = CustomDataset(train_pipeline, img_dir=train_dir,
                                 ann_dir = mask_dir, split= train_split);

    valDataSet = CustomDataset(val_pipeline, img_dir=val_dir,
                                 ann_dir = mask_dir,split=val_split);


    train_loader = build_dataloader(trainDataSet, samples_per_gpu=opt.batch_size,
                                    workers_per_gpu=1,
                                    num_gpus=1, dist=False, shuffle=True);

    val_loader = build_dataloader(valDataSet, samples_per_gpu=opt.val_batch_size,
                                    workers_per_gpu=1,
                                    num_gpus=1, dist=False, shuffle=True);

    return train_loader, val_loader




