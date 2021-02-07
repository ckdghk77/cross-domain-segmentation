'''

This code mainly brought from the repository of https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
and customized to the segmentation task

Thanks to @junyanz
'''

import numpy as np
import os
import sys
import ntpath
import time
from subprocess import Popen, PIPE
from utils.util import tensor2im
from PIL import Image
from random import *

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.exp_name
        self.port = opt.display_port
        self.disp_env = opt.display_env;

        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_folder, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def vis_scalar(self, name, x, y, opts=None):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]

        default_opts = {'title': self.name + "mIOU"}
        if opts is not None:
            default_opts.update(opts)

        try:
            self.vis.line(X=x, Y=y, opts=default_opts, update='append', win=self.display_id+2)
        except VisdomExceptionBase:
            self.create_visdom_connections()


    def vis_image(self, name, img, env=None, opts=None):
        """ vis image in visdom
        """

        name = "[%s]" % str(self.display_id) + name
        default_opts = {'title': name + "seg_result"}
        if opts is not None:
            default_opts.update(opts)

        self.vis.image(img=img, win=self.display_id + 1, opts=opts, env=self.disp_env)

    def vis_table(self, name, tbl, opts=None):

        tbl_str = "<table width=\"100%\"> "
        tbl_str += "<tr> \
                 <th>Term</th> \
                 <th>Value</th> \
                 </tr>"
        for k, v in tbl.items():
            tbl_str += "<tr> \
                       <td>%s</td> \
                       <td>%s</td> \
                       </tr>" % (k, v)

        tbl_str += "</table>"

        default_opts = {'title': name}
        if opts is not None:
            default_opts.update(opts)

        self.vis.text(tbl_str, win=self.display_id + 3, opts=default_opts)


    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def display_current_results(self, visuals, val_score, epoch):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals - tuples of input, pred_segmentation, target_segmentation
            val_score - - mIOU value
        """
        ori_data, seg_pred, seg_target = visuals;

        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols

            try:
                self.vis_scalar("[Val] Overall Acc", epoch, val_score['Overall Acc'])
                self.vis_scalar("[Val] Mean IoU", epoch, val_score['Mean IoU'])
                self.vis_table("[Val] Class IoU", val_score['Class IoU'])
                sample_idx = 1
                target_idx = randint(0,len(ori_data));

                for (img, target, lbl) in zip(ori_data, seg_target, seg_pred):
                    #img = (img.numpy() * 255).astype(np.uint8)

                    pim = Image.open(img);
                    np_img = np.asarray(pim).transpose(2,0,1);

                    if target_idx == sample_idx -1 :
                        break;

                    concat_img = np.concatenate((np_img, target, lbl), axis=2)  # concat along width
                    self.vis_image('Sample %d' % sample_idx , concat_img)
                    sample_idx+=1;
            except VisdomExceptionBase:

                self.create_visdom_connections()



    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k].cpu().item() for k in self.plot_data['legend']])
        #print([losses[k] for k in self.plot_data['legend']])
        try:
            #print(self.plot_data['legend'])
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

