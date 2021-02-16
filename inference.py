
from utils.parser_util import test_learning_args
from utils.parser_util import pretrained_model_parser

import os

from modules.segmentation import Segmentation_model
from utils.data_loader import MMCVTestLoader
from utils.util import *
from utils.visualizer import Visualizer



###################################
 # Parse arguments
###################################
args = test_learning_args().parse_args()
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


###########################################
# Load model from mmsegmentation (https://github.com/open-mmlab/mmsegmentation);
###########################################
pretrained_check_point, config = pretrained_model_parser(args.model_name, args.checkpoints_pt_dir,
                                              args.config_dir);

segm_model = Segmentation_model(pretrained_check_point, config, args);

###########################################
# get DataLoader
###########################################
test_loader = MMCVTestLoader(segm_model.test_load_pipeline,
                                            args);

#segm_model.model.load_state_dict(torch.load(args.checkpoints_folder +
#                                                "/model_latest.pth"))


def test() :
    segm_model.eval()

    input_list = list();
    preds_list = list();

    with torch.no_grad() :
        for iters, data in enumerate(test_loader):

            input, preds = segm_model.test(data);

            input = input[0]
            preds = preds[0]

            # data => donormalize
            input = segm_model.denormalize(input);

            input_list.append(input.cpu().data.numpy())
            preds_list.append(preds);

            if iters >= 5 :
                break;

        preds_list = [segm_model.color_seg(seg) for seg in preds_list]

        save_test_img(input_list, preds_list);


test()

