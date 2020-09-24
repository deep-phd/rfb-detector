"""
This code is the main training code.
"""
import itertools
import logging
import os
import sys
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, ConcatDataset

from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config.fd_config import define_img_size
from vision.utils.misc import Timer, freeze_net_layers, store_labels
from vision.config.config import parser

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

input_img_size = args.input_size  # define input size ,default optional(128/160/320/480/640/1280)
define_img_size(input_img_size)  # must put define_img_size() before 'import fd_config'

if args.loss_type=='focal':
    logging.info("Use focal classification loss.")
elif args.loss_type=='ce':
    logging.info("Use cross entropy calssification loss.")
else:
    logging.fatal("The loss type is wrong.")
    parser.print_help(sys.stderr)
    sys.exit(1)

from vision.ssd.config import fd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.ssd import MatchPrior

cuda_used = 'cuda:' + args.cuda_index
DEVICE = torch.device(cuda_used if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")

def lr_poly(base_lr, iter):
    return base_lr * ((1 - float(iter) / args.num_epochs) ** (args.power))

def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    last_lr = optimizer.param_groups[0]['lr']
    lr = lr_poly(args.lr, i_iter)
    optimizer.param_groups[0]['lr'] = lr
    logging.info('lr change from {} to {}\n'.format(last_lr, lr))

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        print(".", end="", flush=True)
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            print(".", flush=True)
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )

            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

def train_network(dataset_path,model_path,net_type):
    args.datasets = dataset_path
    args.validation_dataset = dataset_path
    args.checkpoint_folder = model_path
    args.log_dir = os.path.join(args.checkpoint_folder,'log')
    args.net = net_type

    timer = Timer()

    logging.info(args)
    if args.net == 'slim':
        create_net = create_mb_tiny_fd
        config = fd_config
    elif args.net == 'RFB':
        create_net = create_Mb_Tiny_RFB_fd
        config = fd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, args.overlap_threshold)

    test_transform = TestTransform(config.image_size, config.image_mean_test, config.image_std)

    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    logging.info("Prepare training datasets.")
    datasets = []

    # voc datasets
    dataset = VOCDataset(dataset_path, transform=train_transform,
                            target_transform=target_transform)
    label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
    store_labels(label_file, dataset.class_names)
    num_classes = len(dataset.class_names)
    print('num_classes: ', num_classes)

    logging.info(f"Stored labels into file {label_file}.")
    # train_dataset = ConcatDataset(datasets)
    train_dataset = dataset
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)

    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")
    net = create_net(num_classes)

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    # add multigpu_train
    if torch.cuda.device_count() >= 1:
        cuda_index_list = [int(v.strip()) for v in args.cuda_index.split(",")]
        net = nn.DataParallel(net, device_ids=cuda_index_list)
        logging.info("use gpu :{}".format(cuda_index_list))

    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    params = [
        {'params': net.module.base_net.parameters(), 'lr': base_net_lr},
        {'params': itertools.chain(
            net.module.source_layer_add_ons.parameters(),
            net.module.extras.parameters()
        ), 'lr': extra_layers_lr},
        {'params': itertools.chain(
            net.module.regression_headers.parameters(),
            net.module.classification_headers.parameters()
        )}
    ]

    net.to(DEVICE)
    criterion = MultiboxLoss(config.priors, iou_threshold=args.iou_threshold, neg_pos_ratio=5,
                             center_variance=0.1, size_variance=0.2, device=DEVICE,
                             num_classes=num_classes, loss_type=args.loss_type)
    if args.optimizer_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params, lr=args.lr)
        logging.info("use Adam optimizer")
    else:
        logging.fatal(f"Unsupported optimizer: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")
    if args.optimizer_type != "Adam":
        if args.scheduler == 'multi-step':
            logging.info("Uses MultiStepLR scheduler.")
            milestones = [int(v.strip()) for v in args.milestones.split(",")]
            scheduler = MultiStepLR(optimizer, milestones=milestones,
                                    gamma=0.1, last_epoch=last_epoch)
        elif args.scheduler == 'poly':
            logging.info("Uses PolyLR scheduler.")
        else:
            logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
            parser.print_help(sys.stderr)
            sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        if args.optimizer_type != "Adam":
            if args.scheduler != "poly":
                if epoch != 0:
                    scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        if args.scheduler == "poly":
            adjust_learning_rate(optimizer, epoch)
        logging.info("epoch: {} lr rate :{}".format(epoch, optimizer.param_groups[0]['lr']))

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            logging.info("validation epoch: {} lr rate :{}".format(epoch, optimizer.param_groups[0]['lr']))
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss:.4f}.pth")
            net.module.save(model_path)
            logging.info(f"Saved model {model_path}")

if __name__ == '__main__':
    train_network(args.datasets, args.checkpoint_folder, args.net)
