
from utils.parser_util import transfer_learning_args
from utils.parser_util import model_parser

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
    if args.continue_training :
        raise("Cannot find model; turn off continue_training flag")
    os.mkdir(checkpoints_folder);

###################################
 # create visdom visualizer
###################################
visualizer = Visualizer(args)   # create a visualizer that display/save images and plots


###########################################
# Load model from mmsegmentation (https://github.com/open-mmlab/mmsegmentation);
###########################################
pretrained_check_point, config = model_parser(args.model_name, args.checkpoints_pt_dir,
                                              args.config_dir);

segm_model = Segmentation_model(pretrained_check_point, config, args);

###########################################
# get DataLoader
###########################################
train_loader, val_loader, trainDataSet, valDataSet = MMCVDataLoader(segm_model.train_load_pipeline,
                                                                    segm_model.test_load_pipeline,
                                                                    args);


best_score = 0.0
cur_epoch = 0;

def train_loop(epoch) :

    segm_model.train()
    interval_loss = 0.0;
    interval_acc = 0.0;
    total_dat_size = len(train_loader);

    for iters, data in enumerate(train_loader):

        losses = segm_model.update(data);

        segment_loss = torch.stack([losses[loss] for loss in losses if "loss" in loss]).mean();
        segment_acc = torch.stack([losses[loss] for loss in losses if "acc" in loss]).mean();

        print(losses['decode.loss_seg'])

        interval_loss += segment_loss.detach().cpu().numpy();
        interval_acc += segment_acc.detach().cpu().numpy();

        if iters % args.print_freq ==0 and iters>0 :
            print("Epoch %d Itrs %d, Loss=%f, Acc=%f" %
                  (epoch, iters, interval_loss/args.print_freq, interval_acc/args.print_freq))
            visualizer.plot_current_losses(epoch,
                                           (iters*args.batch_size)/total_dat_size,
                                           losses)

            interval_loss = 0.0;
            interval_acc = 0.0;

        validate(epoch) # for debugging
        if iters % args.display_freq == 0 and iters > 0 :
            validate(epoch)


    #segm_model.scheduler.step();


def validate(epoch) :
    metrics.reset()
    segm_model.eval()

    denorm_list = list();
    preds_list = list();
    target_list = list();
    input_path_list = list();

    with torch.no_grad() :
        for iters, data in enumerate(val_loader):

            preds, target = segm_model.evaluate(data);

            input_path = os.path.join(
                "datasets",
                args.exp_name,
                "val",
                data['img_metas'][0].data[0][0]['ori_filename']);

            preds = preds[0]
            target = target[0]

            preds = trainDataSet.decode_voc_target(preds).transpose(2,0,1).astype(np.uint8);

            target = trainDataSet.decode_voc_target(target).transpose(2,0,1).astype(np.uint8);

            metrics.update(target, preds);
            # data => donormalize
            #denorm_input = segm_model.denormalize(input);

            if iters < 50 :
                #denorm_list.append(denorm_input);
                input_path_list.append(input_path)
                preds_list.append(preds);
                target_list.append(target);

        val_score = metrics.get_results()

        visualizer.display_current_results((input_path_list, preds_list, target_list), val_score, epoch)
        print(val_score)



saved_epoch = cur_epoch;

for epoch in range(saved_epoch, args.epochs) :
    train_loop(epoch)
    segm_model.save_model('latest');
















