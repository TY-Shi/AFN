r""" AFN training code """
import argparse
import os
import torch.optim as optim
import torch.nn as nn
import torch

from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import config
from common import utils
from data.dataset import CSDataset
from models import create_model
import csv


def get_parser():
    parser = argparse.ArgumentParser(description='Affinity Feature Strengthening for Vessel Segmentation')
    parser.add_argument('--config', type=str, default='config/AFN.yaml', help='Model config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg


def create_csv(path, csv_head):
    # path = "aa.csv"
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        # csv_head = ["good","bad"]
        csv_write.writerow(csv_head)


def write_csv(path, data_row):
    # path  = "aa.csv"
    with open(path, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        # data_row = ["1","2"]
        csv_write.writerow(data_row)


def main():
    global args
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    Logger.initialize(args, training=True)

    # Model initialization
    model = create_model(args)

    Logger.info("=> creating model ...")
    Logger.info("Classes: {}".format(args.classes))
    Logger.log_params(model)

    # Device setup
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = model.cuda()
        model = nn.DataParallel(model)
        Logger.info('Use GPU Parallel.')
    elif torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model

    # Helper classes (for training) initialization
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.base_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
        # optim.Adam([{"params": model.parameters(), "lr": args.base_lr, "weight_decay": args.weight_decay}])
        print('Optimizer: Adam')
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        print('Optimizer: SGD')

    if args.lr_update:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.gamma)
    else:
        scheduler = None

    if args.weight:
        if os.path.isfile(args.weight):
            Logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            Logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            Logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            Logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            Logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            Logger.info("=> no checkpoint found at '{}'".format(args.resume))

    Evaluator.initialize()

    # Dataset initialization
    CSDataset.initialize(datapath=args.datapath)
    dataloader_trn = CSDataset.build_dataloader(args.benchmark,
                                                args.batch_size,
                                                args.nworker,
                                                'train',
                                                args.img_mode,
                                                args.img_size)
    dataloader_val = CSDataset.build_dataloader(args.benchmark,
                                                args.batch_size_val,
                                                args.nworker,
                                                'val',
                                                'same',
                                                None)

    # Train JTFN
    best_val_f1 = float('-inf')
    best_val_pr = float('-inf')
    best_val_r = float('-inf')
    best_val_loss = float('inf')
    val_score_path = os.path.join('logs', args.logname + '.log') + '/' + 'val_[retrain)f1.csv'
    csv_head = ["epoch", "total_loss", "f1", "pr", "recall", "quality", "cor", "com"]
    create_csv(val_score_path, csv_head)
    for epoch in range(args.start_epoch, args.epochs):

        trn_loss_dict, trn_f1, _, _, trn_quality, _, _ = train(epoch, model, dataloader_trn, optimizer, scheduler)

        if args.evaluate:
            with torch.no_grad():
                val_loss_dict, val_f1, val_pr, val_r, val_quality, val_cor, val_com = evaluate(epoch, model,
                                                                                               dataloader_val)
                data_row_f1score = [str(epoch), str(val_loss_dict['total_loss']), str(val_f1), str(val_pr), str(val_r),
                                    str(val_quality), str(val_cor), str(val_com)]
                write_csv(val_score_path, data_row_f1score)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                Logger.save_model_f1(model, epoch, val_f1, optimizer)
            if val_f1 >= best_val_f1 and val_pr >= best_val_pr and val_r >= best_val_r:
                best_val_f1 = val_f1
                best_val_pr = val_pr
                best_val_r = val_r
                Logger.save_model_all(model, epoch, val_f1, val_pr, val_r, optimizer)

        for key in trn_loss_dict.keys():
            Logger.tbd_writer.add_scalars('data/loss_train', {'trn_loss_' + str(key): trn_loss_dict[key]}, epoch)
        for key in val_loss_dict.keys():
            Logger.tbd_writer.add_scalars('data/loss_train_val', {'trn_loss_' + str(key): trn_loss_dict[key],
                                                                  'val_loss_' + str(key): val_loss_dict[key]}, epoch)
        Logger.tbd_writer.add_scalars('data/f1', {'trn_f1': trn_f1, 'val_f1': val_f1}, epoch)
        Logger.tbd_writer.flush()
    print('Best F1: ', best_val_f1)
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')


def train(epoch, model, dataloader, optimizer, scheduler):
    r""" Train AFN """
    if torch.cuda.device_count() > 1:
        model.module.train_mode()
    else:
        model.train_mode()
    average_meter = AverageMeter(dataloader.dataset)

    # max_iter = args.epochs * len(dataloader)
    for idx, batch in enumerate(dataloader):

        # 1. Forward pass
        batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch
        output_dict = model(batch)
        out = output_dict['output']
        pred_mask = torch.where(out >= 0.5, 1, 0)

        # 2. Compute loss & update model parameters
        loss_dict = model.module.compute_objective(output_dict,
                                                   batch) if torch.cuda.device_count() > 1 else model.compute_objective(
            output_dict, batch_dict=batch)

        loss = loss_dict['total_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.lr_update:
            scheduler.step()

        # 3. Evaluate prediction
        f1, pr, r, quality, cor, com = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(f1, pr, r, quality, cor, com, loss_dict)
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Training', epoch)
    avg_loss_dict = dict()
    for key in average_meter.loss_buf.keys():
        avg_loss_dict[key] = utils.mean(average_meter.loss_buf[key])
    f1 = average_meter.compute_f1()
    pr = average_meter.compute_precision()
    r = average_meter.compute_recall()
    quality = average_meter.compute_quality()
    cor = average_meter.compute_correctness()
    com = average_meter.compute_completeness()

    return avg_loss_dict, f1, pr, r, quality, cor, com


def evaluate(epoch, model, dataloader):
    r""" Eval AFN """
    # Force randomness during training / freeze randomness during testing
    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        # 1. Forward pass
        batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch
        output_dict = model(batch)
        out = output_dict['output']
        pred_mask = torch.where(out >= 0.5, 1, 0)

        # 2. Compute loss & update model parameters
        loss_dict = model.module.compute_objective(output_dict,
                                                   batch) if torch.cuda.device_count() > 1 else model.compute_objective(
            output_dict, batch_dict=batch)

        # 3. Evaluate prediction
        f1, pr, r, quality, cor, com = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(f1, pr, r, quality, cor, com, loss_dict)
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=10)

    # Write evaluation results
    average_meter.write_result('Validation', epoch)
    avg_loss_dict = dict()
    for key in average_meter.loss_buf.keys():
        avg_loss_dict[key] = utils.mean(average_meter.loss_buf[key])
    f1 = average_meter.compute_f1()
    pr = average_meter.compute_precision()
    r = average_meter.compute_recall()
    quality = average_meter.compute_quality()
    cor = average_meter.compute_correctness()
    com = average_meter.compute_completeness()

    return avg_loss_dict, f1, pr, r, quality, cor, com


if __name__ == '__main__':
    main()
