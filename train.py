"""Main training module

Includes loading application argument, initialize global variables and running the train / eval session.
"""

import argparse
import time
import os
from noisy_dataset import DatasetGenerator
from tqdm import tqdm
from utils import *
from loss import SCELoss
from models import *

# Global steps - used in logging
GLOBAL_STEP, EVAL_STEP, EVAL_BEST_ACC, EVAL_BEST_ACC_TOP5 = 0, 0, 0, 0
# Initialization params - used all over the module
args, logger, device = None, None, None


def get_args():
    """Parsing module args"""

    parser = argparse.ArgumentParser(description='SCE Loss')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2_reg', type=float, default=5e-4)
    parser.add_argument('--grad_bound', type=float, default=5.0)
    parser.add_argument('--train_log_every', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_path', default='../../datasets', type=str)
    parser.add_argument('--data_nums_workers', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--loss', type=str, default='SCE', help='SCE, CE')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha scale')
    parser.add_argument('--beta', type=float, default=1.0, help='beta scale')
    parser.add_argument('--version', type=str, default='SCE_stable_v1', help='Version')
    parser.add_argument('--model', type=str, default='scenet', help='Name of the model architecture to be used')
    parser.add_argument('--optim', type=str, default='sgd', help='Name of the model optimizer to be used')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--label_weight', type=bool, default=False, help='Using label weight (True) or not (False)')

    return parser.parse_args()


def init():
    """Initialization of module global variables"""

    # parse global args
    global args
    global logger
    global device
    args = get_args()
    # Make log configurations
    if not os.path.exists('logs'):
        os.makedirs('logs')
    log_file_name = os.path.join('logs', args.version + '.log')
    logger = setup_logger(name=args.version, log_file=log_file_name)
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))

    # Decide on torch device
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda')
        logger.info("Using CUDA!")
    else:
        device = torch.device('cpu')


def model_eval(epoch, fixed_cnn, data_loader):
    """ Evaluating the training process on current model """

    global EVAL_STEP
    fixed_cnn.eval()
    valid_loss_meters = AverageMeter()
    valid_acc_meters = AverageMeter()
    valid_acc5_meters = AverageMeter()
    ce_loss = torch.nn.CrossEntropyLoss()

    for images, labels in tqdm(data_loader["test_dataset"]):
        start = time.time()
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            pred = fixed_cnn(images)
            loss = ce_loss(pred, labels)
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))

        valid_loss_meters.update(loss.item())
        valid_acc_meters.update(acc.item())
        valid_acc5_meters.update(acc5.item())
        end = time.time()

        EVAL_STEP += 1
        if EVAL_STEP % args.train_log_every == 0:
            display = log_display(epoch=epoch,
                                  global_step=GLOBAL_STEP,
                                  time_elapse=end-start,
                                  loss=loss.item(),
                                  test_loss_avg=valid_loss_meters.avg,
                                  acc=acc.item(),
                                  test_acc_avg=valid_acc_meters.avg,
                                  test_acc_top5_avg=valid_acc5_meters.avg)
            logger.info(display)
    display = log_display(epoch=epoch,
                          global_step=GLOBAL_STEP,
                          time_elapse=end-start,
                          loss=loss.item(),
                          test_loss_avg=valid_loss_meters.avg,
                          acc=acc.item(),
                          test_acc_avg=valid_acc_meters.avg,
                          test_acc_top5_avg=valid_acc5_meters.avg)
    logger.info(display)
    return valid_acc_meters.avg, valid_acc5_meters.avg


def update_target_prob(pred, indexes, trainset):
    """My try to add label weights according to model prediction. Only reduced performance
    Should not be used by default
    """
    pred_arr = pred.cpu().data.numpy()
    indexes_arr = indexes.cpu().data.numpy()
    threshold = 0.3
    highest_pred = np.argmax(pred_arr, axis=1)
    for idx, high_pred in enumerate(highest_pred):
        if np.where(pred_arr[idx][high_pred] >= threshold, True, False) and \
                list(trainset.targets[indexes_arr[idx]].values())[high_pred] != 0:
            key = list(trainset.targets[indexes_arr[idx]].keys())[high_pred]
            labels_idx = np.where(np.array(list(trainset.targets[indexes_arr[idx]].values())) != 0)[0]
            if labels_idx[0] == high_pred:
                other_key = list(trainset.targets[indexes_arr[idx]].keys())[labels_idx[1]]
                trainset.targets[indexes_arr[idx]][key] *= 1.02
                trainset.targets[indexes_arr[idx]][other_key] *= 0.98
            else:
                other_key = list(trainset.targets[indexes_arr[idx]].keys())[labels_idx[0]]
                trainset.targets[indexes_arr[idx]][key] *= 1.02
                trainset.targets[indexes_arr[idx]][other_key] *= 0.98
            sum_target = trainset.targets[indexes_arr[idx]][key] + trainset.targets[indexes_arr[idx]][other_key]
            trainset.targets[indexes_arr[idx]][key] = trainset.targets[indexes_arr[idx]][key] / sum_target
            trainset.targets[indexes_arr[idx]][other_key] = trainset.targets[indexes_arr[idx]][other_key] / sum_target


def train_fixed(starting_epoch, data_loader, fixed_cnn, criterion, fixed_cnn_optmizer, fixed_cnn_scheduler):
    """ Training function """

    global GLOBAL_STEP, EVAL_BEST_ACC, EVAL_STEP, EVAL_BEST_ACC_TOP5

    for epoch in tqdm(range(starting_epoch, args.epoch)):
        logger.info("=" * 20 + "Training" + "=" * 20)
        fixed_cnn.train()
        train_loss_meters = AverageMeter()
        train_acc_meters = AverageMeter()
        train_acc5_meters = AverageMeter()

        for images, labels, indexes in tqdm(data_loader["train_dataset"]):
            images, labels = images.to(device), labels.to(device)
            start = time.time()
            fixed_cnn.zero_grad()
            fixed_cnn_optmizer.zero_grad()
            pred = fixed_cnn(images)
            if args.label_weight:
                update_target_prob(pred, indexes, data_loader["train_dataset"].dataset)
            loss = criterion(pred, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(fixed_cnn.parameters(), args.grad_bound)
            fixed_cnn_optmizer.step()
            acc, acc5 = accuracy(pred, labels, topk=(1, 5))
            acc_sum = torch.sum((torch.max(pred, 1)[1] == labels).type(torch.float))
            total = pred.shape[0]
            acc = acc_sum / total

            train_loss_meters.update(loss.item())
            train_acc_meters.update(acc.item())
            train_acc5_meters.update(acc5.item())

            end = time.time()

            GLOBAL_STEP += 1
            if GLOBAL_STEP % args.train_log_every == 0:
                lr = fixed_cnn_optmizer.param_groups[0]['lr']
                display = log_display(epoch=epoch,
                                      global_step=GLOBAL_STEP,
                                      time_elapse=end-start,
                                      loss=loss.item(),
                                      loss_avg=train_loss_meters.avg,
                                      acc=acc.item(),
                                      acc_top1_avg=train_acc_meters.avg,
                                      acc_top5_avg=train_acc5_meters.avg,
                                      lr=lr,
                                      gn=grad_norm)
                logger.info(display)

        fixed_cnn_scheduler.step()

        # Logging
        logger.info("="*20 + "Eval" + "="*20)
        curr_acc, curr_acc5 = model_eval(epoch, fixed_cnn, data_loader)
        logger.info("curr_acc\t%.4f" % curr_acc)
        logger.info("BEST_ACC\t%.4f" % EVAL_BEST_ACC)
        logger.info("curr_acc_top5\t%.4f" % curr_acc5)
        logger.info("BEST_ACC_top5\t%.4f" % EVAL_BEST_ACC_TOP5)

        EVAL_BEST_ACC = max(curr_acc, EVAL_BEST_ACC)
        EVAL_BEST_ACC_TOP5 = max(curr_acc5, EVAL_BEST_ACC_TOP5)


def run_experiment():
    """Entry point for running the train / eval process"""

    # Dataset
    dataset = DatasetGenerator(batch_size=args.batch_size,
                               data_path=args.data_path,
                               num_of_workers=args.data_nums_workers,
                               seed=args.seed)

    data_loader = dataset.get_loader()

    num_classes = 10

    # Load model dynamically
    fixed_cnn = load_model(args.model)
    fixed_cnn = torch.nn.DataParallel(fixed_cnn)
    fixed_cnn.to(device)

    if args.loss == 'SCE':
        criterion = SCELoss(alpha=args.alpha, beta=args.beta, num_classes=num_classes)
    elif args.loss == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        logger.info("Unknown loss")

    logger.info(f'Criterion : {criterion.__class__.__name__}')
    logger.info(f'Number of Trainable Parameters %.4f' % count_parameters_in_MB(fixed_cnn))

    fixed_cnn_optim = get_optimizer(args.optim, fixed_cnn, args)

    fixed_cnn_scheduler = torch.optim.lr_scheduler.MultiStepLR(fixed_cnn_optim, milestones=[40, 80], gamma=0.1)
    starting_epoch = 0
    train_fixed(starting_epoch, data_loader, fixed_cnn, criterion, fixed_cnn_optim, fixed_cnn_scheduler)


if __name__ == '__main__':
    # Init global args, logger, device
    init()
    # Start training process
    run_experiment()
