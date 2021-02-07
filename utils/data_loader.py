
import os

import numpy as np

from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.builder import build_dataloader


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class MyCustomDataSet(CustomDataset) :

    cmap = voc_cmap()
    def __init__(self,pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        super(MyCustomDataSet, self).__init__(pipeline,
                 img_dir,
                 img_suffix=img_suffix,
                 ann_dir=ann_dir,
                 seg_map_suffix=seg_map_suffix,
                 split=split,
                 data_root=data_root,
                 test_mode=test_mode,
                 ignore_index=ignore_index,
                 reduce_zero_label=reduce_zero_label,
                 classes=classes,
                 palette=palette)

    @classmethod
    def decode_voc_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


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


    trainDataSet = MyCustomDataSet(train_pipeline, img_dir=train_dir,
                                 ann_dir = mask_dir, split= train_split);

    valDataSet = MyCustomDataSet(val_pipeline, img_dir=val_dir,
                                 ann_dir = mask_dir,split=val_split);


    train_loader = build_dataloader(trainDataSet, samples_per_gpu=opt.batch_size,
                                    workers_per_gpu=1,
                                    num_gpus=1, dist=False, shuffle=True);

    val_loader = build_dataloader(valDataSet, samples_per_gpu=opt.val_batch_size,
                                    workers_per_gpu=1,
                                    num_gpus=1, dist=False, shuffle=True);

    return train_loader, val_loader, trainDataSet, valDataSet




