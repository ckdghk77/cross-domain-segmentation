
from utils.parser_util import transfer_learning_args
from utils.parser_util import pretrained_model_parser

import os

from modules.segmentation import Segmentation_model
from utils.data_loader import MMCVDataLoader
from utils.util import *
from utils.visualizer import Visualizer



###################################
 # Parse arguments
###################################
args = transfer_learning_args().parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed);
torch.manual_seed(args.seed);
if args.cuda :
    torch.cuda.manual_seed(args.seed)

###################################
 # create folder
###################################
checkpoints_folder = '{}/exp_{}_on_{}'.format(args.checkpoints_dir, args.model_name, args.exp_name)
args.checkpoints_folder = checkpoints_folder

if not os.path.exists(checkpoints_folder) :
    if args.continue_train :
        raise("Cannot find model; turn off continue_training flag")
    os.mkdir(checkpoints_folder);


###################################
 # create visdom visualizer
###################################
visualizer = Visualizer(args)   # create a visualizer that display/save images and plots


###########################################
# Load model from mmsegmentation (https://github.com/open-mmlab/mmsegmentation);
###########################################
pretrained_check_point, config = pretrained_model_parser(args.model_name, args.checkpoints_pt_dir,
                                              args.config_dir);
'''
if args.continue_train :
    load_check_point = args.checkpoints_folder + "/model_latest.pth";
else :
    load_check_point = pretrained_check_point
'''
segm_model = Segmentation_model(pretrained_check_point , config, args);

if args.continue_train :
    segm_model.model.load_state_dict(torch.load(args.checkpoints_folder +
                                                "/model_latest.pth"));


###########################################
# get DataLoader
###########################################
train_loader, val_loader = MMCVDataLoader(segm_model.train_load_pipeline,
                                            segm_model.val_load_pipeline,
                                            args);

cur_epoch = 0;

def train_loop(epoch) :

    segm_model.train()
    interval_loss = 0.0;
    interval_acc = 0.0;

    for iters, data in enumerate(train_loader):

        losses = segm_model.update(data);

        segment_loss = torch.stack([losses[loss] for loss in losses if "loss" in loss]).mean();
        segment_acc = torch.stack([losses[loss] for loss in losses if "acc" in loss]).mean();

        #print(losses['decode.loss_seg'])

        interval_loss += segment_loss.detach().cpu().numpy();
        interval_acc += segment_acc.detach().cpu().numpy();

        if iters % args.print_freq ==0 and iters>0 :
            print("Epoch %d Itrs %d, Loss=%f, Acc=%f" %
                  (epoch, iters, interval_loss/args.print_freq, interval_acc/args.print_freq))

            interval_loss = 0.0;
            interval_acc = 0.0;

        if iters % args.display_freq == 0 and iters > 0 :

            validate(epoch, iters/len(train_loader))


def validate(epoch, iter_ratio) :
    segm_model.eval()

    input_list = list();
    preds_list = list();
    target_list = list();

    with torch.no_grad() :
        for iters, data in enumerate(val_loader):

            input, preds, target = segm_model.validate(data);

            input = input[0]
            preds = preds[0]
            target = target[0]

            # data => donormalize
            input = segm_model.denormalize(input);

            input_list.append(input.cpu().data.numpy())
            preds_list.append(preds);
            target_list.append(target.cpu().data.numpy());

        val_score = segm_model.evaluate_mIoU(preds_list,
                                        target_list);

        preds_list = [segm_model.color_seg(seg) for seg in preds_list]
        target_list = [segm_model.color_seg(seg) for seg in target_list]


        visualizer.display_current_results((input_list, preds_list, target_list),
                                           val_score,
                                           epoch,
                                           iter_ratio,
                                           segm_model.class_names)

saved_epoch = cur_epoch;

for epoch in range(saved_epoch, args.epochs) :
    segm_model.save_model('latest');
    train_loop(epoch)
    segm_model.save_model('latest');

