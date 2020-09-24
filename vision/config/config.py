import argparse

def str2bool(s):
    return s.lower() in ('true', '1')

parser = argparse.ArgumentParser(description='train With Pytorch')

parser.add_argument('--datasets', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--net', default="slim", help="The network architecture ,optional(RFB , slim)")

# Params for loss 
parser.add_argument('--loss_type', default='ce', type=str,
        help='loss fun for classification, default options: ce/focal')

# Params for SGD
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')

parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('--base_net_lr', default='0.0001', type=float, help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float, help='initial learning rate for the layers not in base net and prediction heads.')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
        help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Train params
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_epochs', default=100, type=int, help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=2, type=int, help='the number epochs')
parser.add_argument('--debug_steps', default=50, type=int, help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool, help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/', help='Directory for saving checkpoint models')
parser.add_argument('--log_dir', default='./models/logs', help='lod dir')
parser.add_argument('--cuda_index', default="0", type=str,
                    help='Choose cuda index.If you have 4 GPUs, you can set it like 0,1,2,3')
parser.add_argument('--power', default=2, type=int, help='poly lr pow')
parser.add_argument('--overlap_threshold', default=0.34999999404, type=float,
                    help='overlap_threshold')
parser.add_argument('--iou_threshold', default=0.34999999404, type=float,
                    help='iou_threshold')
parser.add_argument('--optimizer_type', default="SGD", type=str,
                    help='optimizer_type')
parser.add_argument('--input_size', default=640, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')

