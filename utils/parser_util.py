import argparse
import os

def transfer_learning_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')  ##  59 good
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.c')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate.')
    parser.add_argument('--lr-decay', type=int, default=200,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR decay factor.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Number of samples per batch.')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='Number of samples per val-batch.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers.')


    parser.add_argument('--exp-name', type=str, default="voc2012_toon_725586",
                        help='exp-name)')
    parser.add_argument('--model-name', type=str, default="deeplabv3plus_r50_voc12aug",
                        help='{}_{}_{}.format(base, backbone, trained_dataset'
                             'e.g. base : {deeplabv3plus},'
                             '     backbone : {r101, r18, r50},'
                             '     trained_dataset : {cityscapes, voc2012, voc2012aug}'
                             'you must download pretrained model from the mmsegmentation github page and'
                             'place it in checkpoints_pretrained')
    parser.add_argument('--continue_training', action='store_true');
    parser.add_argument('--gpu-id', type=str, default="0", help="GPU ID");
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop size')


    parser.add_argument('--checkpoints_pt_dir', type=str, default='./checkpoints_pretrained',
                        help='pretrained segmentation models should be saved here')
    parser.add_argument('--config_dir', type=str, default='./configs',
                        help='you must place configs from mmsegmentation github page')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--result_dir', type=str, default='./result', help='result file name here')


    #visdom option
    parser.add_argument('--print_freq', type=int, default=50, help='frequency of print losses on console')
    parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=4,
                        help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_server', type=str, default="http://localhost",
                        help='visdom server of the web display')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size for visdom')
    parser.add_argument('--display_env', type=str, default='main',
                        help='visdom display environment name (default is "main")')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')


    return parser

def model_parser(model_name, checkpoint_pretrained_dir, config_dir):


    checkpoint_map = {
        'deeplabv3plus_r101_voc12aug' : "deeplabv3plus_r101-d8_512x512_20k_voc12aug_20200617_102345-c7ff3d56.pth",
        'deeplabv3plus_r50_voc12aug' : "deeplabv3plus_r50-d8_512x512_40k_voc12aug_20200613_161759-e1b43aa9.pth"
    }

    config_map = {
        'deeplabv3plus_r101_voc12aug': "deeplabv3plus/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py",
        'deeplabv3plus_r50_voc12aug': "deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py"
    }

    checkpoint_file = os.path.join(checkpoint_pretrained_dir, checkpoint_map[model_name]);
    config_file = os.path.join(config_dir, config_map[model_name]);

    return checkpoint_file, config_file
