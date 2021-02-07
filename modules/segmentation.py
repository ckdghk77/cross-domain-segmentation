
from mmseg.apis import inference_segmentor
from mmseg.apis import init_segmentor
from mmseg.core.evaluation import mean_iou

from mmseg.core.evaluation.class_names import voc_classes

from torch import nn
from mmcv.runner import build_optimizer
from utils.util import resize
import torch.nn.functional as F

from torchvision.transforms.functional import normalize

import os

import numpy as np
import torch
import copy

class Segmentation_model(nn.Module) :

    def __init__(self, checkpoint_file, config_file, args) :
        super(Segmentation_model, self).__init__();

        self.model_name = args.model_name;
        self.model = init_segmentor(config_file, checkpoint_file, device="cpu")
        self.num_classes = self.model.decode_head.num_classes
        self.checkpoint_dir = args.checkpoints_folder

        if args.exp_name.startswith("voc") :
            self.class_names = voc_classes()

        ###########################################
        # Define Optimizer and criterion
        ###########################################
        self.model.cfg.optimizer.lr = args.lr;
        self.optimizer = build_optimizer(self.model, self.model.cfg.optimizer)


        ###########################################
        # Define data loader config
        ###########################################
        self.train_load_pipeline, self.test_load_pipeline =\
            self.process_pipeline(copy.deepcopy(self.model.cfg.data.train.pipeline),
                              copy.deepcopy(self.model.cfg.data.test.pipeline[:]))

        if args.cuda:
            self.model = self.model.cuda()

        # test a single image and show the results
        img = './datasets/voc2012/train/2007_000027.jpg'  # or img = mmcv.imread(img), which will only load it once
        result = inference_segmentor(self.model, img)

        self.model.show_result(img, result, out_file='result.jpg')


    def process_pipeline(self, train_pipe, test_pipe):
        '''
            Add "LoadAnotation and Collect" to test pipeline

        :return:
        '''

        for pipe in train_pipe :
            if pipe.type == "LoadAnnotations" :
                test_pipe.insert(1,pipe);

        for pipe in test_pipe :
            if pipe.type == "Collect" :
                sub_pipe['keys'].append("gt_semantic_seg")
            elif len(pipe) > 1 :
                for sub_pipe in pipe['transforms'] :
                    if sub_pipe.type == "Collect" :
                        sub_pipe['keys'].append("gt_semantic_seg")


        return train_pipe, test_pipe

    def update(self, data):
        # img must be list of file names
        # target also

        self.model.zero_grad()
        self.model.train()

        input = torch.stack(data['img'].data).squeeze();
        target = torch.stack(data['gt_semantic_seg'].data).squeeze();

        if next(self.model.parameters()).is_cuda:
            input, target = input.cuda(), target.unsqueeze(1).cuda()

        loss = self.model(return_loss=True, img = input,
                          img_metas= data['img_metas'].data,
                          gt_semantic_seg = target);

        loss['decode.loss_seg'].backward();

        self.optimizer.step()

        return loss

    def inference(self, data):

        self.model.eval()

        inputs = list(data['img'][0].data);
        targets = data['gt_semantic_seg'][0].data;
        metas = list(data['img_metas'][0].data);

        if next(self.model.parameters()).is_cuda:
            for d_i in range(len(inputs)) :
                inputs[d_i] = inputs[d_i].cuda().unsqueeze(0)

        with torch.no_grad() :
            output = self.model(return_loss=False,rescale=True,
                              img = inputs,
                              img_metas= metas);


        inputs = inputs[0];

        resized_input_list = list()
        resized_target_list = list();               # Rescale to original input size (require modification)
        for t_i in range(targets.shape[0]) :
            origianl_shape = metas[0][t_i]['ori_shape'][:2];
            resized_target_list.append(resize(targets[t_i], origianl_shape));
            resized_input_list.append(resize(inputs[t_i], origianl_shape));

        return resized_input_list, output, resized_target_list

    def evaluate_mIoU(self, pred, target):
        all_acc, acc, dice = mean_iou(results=pred,
                 gt_seg_maps=target,
                 num_classes=self.num_classes,
                 ignore_index=255)

        return all_acc, acc, dice

    def color_seg(self, seg) :
        palette = self.model.PALETTE;
        palette = np.array(palette)

        assert palette.shape[0] == self.num_classes
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        return color_seg.transpose(2,0,1)

    def save_model(self, save_suffix):
        save_filename = 'model_' + save_suffix + '.pth';
        save_path = os.path.join(self.checkpoint_dir, save_filename);

        if next(self.model.parameters()).is_cuda :
            torch.save(self.model.cpu().state_dict, save_path);
        else :
            torch.save(self.model.state_dict, save_path);

    def denormalize(self, tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        for proc in self.test_load_pipeline :
            if proc.type == "Normalize" :
                mean = proc.mean
                std = proc.std;

        mean = np.array(mean)
        std = np.array(std)

        _mean = -mean / std
        _std = 1 / std

        return normalize(tensor, _mean, _std)







