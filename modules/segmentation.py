
from mmseg.apis import init_segmentor

from torch import nn
from mmcv.runner import build_optimizer
from utils.util import add_prefix
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

    def evaluate(self, data):

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


        target_list = list();               # Rescale to original input size (require modification)
        for t_i in range(targets.shape[0]) :

            target_list.append(F.interpolate(targets[t_i].unsqueeze(0).unsqueeze(0),
                                             size=metas[0][t_i]['ori_shape'][:2],
                                             mode='nearest').squeeze());

        return output, target_list

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







