r""" JTFN training code """
import argparse
import os

import torch.nn as nn
import torch

from common.logger import AverageMeter
from common.evaluation import Evaluator
from common import config
from common import utils
from common.show import show_compare
from data.dataset import CSDataset
from models import create_model
import PIL.Image as Image
import csv
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description='JTFN for Curvilinear Structure Segmentation')
    parser.add_argument('--config', type=str, default='config/UNet_DRIVE.yaml', help='Model config file')
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

    # Model initialization
    model = create_model(args)

    print("=> creating model ...")
    print("Classes: {}".format(args.classes))

    # Device setup
    print('# available GPUs: %d' % torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = model.cuda()
        model = nn.DataParallel(model)
        print('Use GPU Parallel.')
    elif torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model
    print("args.weight", args.weight)

    if args.weight:
        if os.path.isfile(args.weight):
            print("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded weight '{}'".format(args.weight))
        else:
            print("=> no weight found at '{}'".format(args.weight))
    else:
        raise RuntimeError("Please support weight.")

    Evaluator.initialize()

    # Dataset initialization
    CSDataset.initialize(datapath=args.datapath)
    dataloader_val = CSDataset.build_dataloader(args.benchmark,
                                                args.batch_size_val,
                                                args.nworker,
                                                'val',
                                                'same',
                                                None)
    # show_weight(model, dataloader_val)
    # threshold_list = np.arange(0.1, 1, 0.01).tolist()
    #print(threshold_list)
    #threshold_list = [0.4, 0.5]
    best_th = 0

    best_f1, best_pr, best_r, best_quality, best_cor, best_com = 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for threshold in range(15, 26):
            thresh = threshold / 100
            val_loss_dict, val_f1, val_pr, val_r, val_quality, val_cor, val_com = evaluate_threshold(model, dataloader_val, thresh)

            if val_f1 > best_f1:
                best_th = thresh
                best_f1, best_pr, best_r, best_quality, best_cor, best_com \
                   = val_f1, val_pr, val_r, val_quality, val_cor, val_com
            print(best_th, best_f1)

        # val_loss_dict, val_f1, val_pr, val_r, val_quality, val_cor, val_com = evaluate(model, dataloader_val)
        # val_loss_dict, val_f1, val_pr, val_r, val_quality, val_cor, val_com = evaluate_threshold(model, dataloader_val, 0.16)

    # print('F1: {:.2f} Precision: {:.2f} Recall: {:.2f}'.format(val_f1.item(), val_pr.item(), val_r.item()))
    # print('Quality: {:.2f} Correctness: {:.2f} Completeness: {:.2f}'.format(val_quality.item(), val_cor.item(),
    #                                                                         val_com.item()))
    print(best_th)
    print('F1: {:.2f} Precision: {:.2f} Recall: {:.2f}'.format(best_f1.item(), best_pr.item(), best_r.item()))
    print('Quality: {:.2f} Correctness: {:.2f} Completeness: {:.2f}'.format(best_quality.item(), best_cor.item(),
                                                                           best_com.item()))
    print('==================== Finished Testing ====================')


def show_weight(model, dataloader):
    path = "save_image/weight/"

    for idx, batch in enumerate(dataloader):
        # 1. Forward pass
        # batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch
        img = batch['img']
        gt = batch['anno_mask']

        weight = np.load("weight_scale.npy")

        print(gt.shape, weight.shape)

        gt_weight_1 = gt * weight[:, 0, :, :]
        gt_weight_2 = gt * weight[:, 1, :, :]
        gt_weight_3 = gt * weight[:, 2, :, :]

        gt_weight_1 = gt_weight_1[0, 0, :, :].cpu().detach().numpy()
        gt_weight_2 = gt_weight_2[0, 0, :, :].cpu().detach().numpy()
        gt_weight_3 = gt_weight_3[0, 0, :, :].cpu().detach().numpy()

        print(gt_weight_1.shape, gt_weight_2.shape, gt_weight_3.shape)

        im = Image.fromarray(gt_weight_1 * 255.).convert("L")
        im_name = path + '{}_weight_1.png'.format(str(idx))
        im.save(im_name)

        im = Image.fromarray(gt_weight_2 * 255.).convert("L")
        im_name = path + '{}_weight_2.png'.format(str(idx))
        im.save(im_name)

        im = Image.fromarray(gt_weight_3 * 255.).convert("L")
        im_name = path + '{}_weight_3.png'.format(str(idx))
        im.save(im_name)

        # print(gt_weight_1.shape, gt_weight_2.shape, gt_weight_3.shape)


def save_image(prob, affinity, out_affinity_1, idx):
    N = prob.shape[0]
    image_path = 'save_image/affuse/prob/'
    aff_path = 'save_image/affuse/affinity/'
    for num in range(0, N):
        image = prob[num, 0, :, :].cpu().detach().numpy()
        # print("image",image.shape)
        # print("unique",np.unique(image))
        im = Image.fromarray(image * 255.).convert("L")
        im_name = image_path + '{}.png'.format(str(idx))
        im.save(im_name)
    for num in range(0, N):
        affinity_image = affinity[num, :, :, :].cpu().detach().numpy()
        Affinity_num = affinity_image.shape[0]
        for direct in range(0, Affinity_num):
            image = affinity_image[direct, :, :]
            # print("image", image.shape)
            print("image_0", np.unique(image))
            im = Image.fromarray(image * 255.).convert("L")
            im_name = aff_path + '{}_aff_{}.png'.format(str(idx), str(direct))
            im.save(im_name)
    for num in range(0, N):
        affinity_image = out_affinity_1[num, :, :, :].cpu().detach().numpy()
        Affinity_num = affinity_image.shape[0]
        for direct in range(0, Affinity_num):
            image = affinity_image[direct, :, :]
            # print("image", image.shape)
            im = Image.fromarray(image * 255.).convert("L")
            im_name = aff_path + '{}_aff_{}_1.png'.format(str(idx), str(direct))
            im.save(im_name)


def evaluate(model, dataloader):
    r""" Eval JTFN """
    global args
    args = get_parser()
    # Force randomness during training / freeze randomness during testing
    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    val_score_path = os.path.join('logs', args.logname + '.log') + '/' + 'single_image_val.csv'
    csv_head = ["image_name", "f1", "pr", "recall", "quality", "cor", "com"]
    create_csv(val_score_path, csv_head)

    for idx, batch in enumerate(dataloader):
        # 1. Forward pass
        batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch
        output_dict = model(batch)
        out = output_dict['output']
        pred_mask = torch.where(out >= 0.5, 1, 0)
        # out_affinity = output_dict['step_' + '2' + '_output_affinity'][0]
        # out_affinity_1 = output_dict['step_' + '2' + '_output_affinity'][1]
        # print(out_affinity.shape, out_affinity_1.shape)
        # save_image(out, out_affinity, out_affinity_1, idx)

        # 2. Compute loss & update model parameters
        loss_dict = model.module.compute_objective(output_dict,
                                                   batch) if torch.cuda.device_count() > 1 else model.compute_objective(
            output_dict, batch_dict=batch)

        # 3. Evaluate prediction
        f1, pr, r, quality, cor, com = Evaluator.classify_prediction(pred_mask.clone(), batch)
        img_name = batch.get('img_name')
        data_row_f1score = [str(img_name), str(f1), str(pr), str(r),
                            str(quality), str(cor), str(com)]
        write_csv(val_score_path, data_row_f1score)
        average_meter.update(f1, pr, r, quality, cor, com, loss_dict)
        # print(f1, pr, r, quality, cor, com)

        #img = batch.get("img")[0, 0, :, :]
        ##pred_mask = pred_mask[0,0,:,:]
        #print(img_name)
        #rgb_img = torch.stack([pred_mask, pred_mask, pred_mask], dim=-1).cpu().numpy()*255.
        ##rgb_img = torch.stack([img, img, img], dim=-1).cpu().numpy()*255.
        #print(rgb_img.shape)
        # print(rgb_img)
        #rgb_img = show_compare(rgb_img, pred_mask, batch.get('anno_mask'),
        #                       batch.get('ignore_mask') if 'ignore_mask' in batch.keys() else None)

        #im = Image.fromarray(np.uint8(rgb_img))
        #im_name = "save_image/topoloss/drive_image/" + '{}.png'.format(str(img_name[0]))
        #im.save(im_name)

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


def evaluate_threshold(model, dataloader, threshold=0.5):
    r""" Eval JTFN """

    # Force randomness during training / freeze randomness during testing
    if torch.cuda.device_count() > 1:
        model.module.eval()
    else:
        model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    val_score_path = os.path.join('logs', args.logname + '.log') + '/' + 'single_image_val.csv'
    csv_head = ["image_name", "f1", "pr", "recall", "quality", "cor", "com"]
    create_csv(val_score_path, csv_head)

    for idx, batch in enumerate(dataloader):
        # 1. Forward pass
        batch = utils.to_cuda(batch) if torch.cuda.is_available() else batch

        # print(batch['img'].shape)

        output_dict = model(batch)
        out = output_dict['output']
        pred_mask = torch.where(out >= threshold, 1, 0)
        #out_affinity = output_dict['step_' + '2' + '_output_affinity'][0]
        #out_affinity_1 = output_dict['step_' + '2' + '_output_affinity'][1]
        #print(out_affinity.shape, out_affinity_1.shape)
        # save_image(out, out_affinity, out_affinity_1, idx)

        # 2. Compute loss & update model parameters
        loss_dict = model.module.compute_objective(output_dict,
                                                   batch) if torch.cuda.device_count() > 1 else model.compute_objective(
            output_dict, batch_dict=batch)

        # 3. Evaluate prediction
        f1, pr, r, quality, cor, com = Evaluator.classify_prediction(pred_mask.clone(), batch)
        img_name = batch.get('img_name')
        data_row_f1score = [str(img_name), str(f1), str(pr), str(r),
                            str(quality), str(cor), str(com)]
        write_csv(val_score_path, data_row_f1score)
        average_meter.update(f1, pr, r, quality, cor, com, loss_dict)
        #
        # img_name = batch.get('img_name')
        #
        # img = batch.get("img")[0, 0, :, :]
        # #pred_mask = pred_mask[0,0,:,:]
        # print(img_name)
        # #rgb_img = torch.stack([pred_mask, pred_mask, pred_mask], dim=-1).cpu().numpy()*255.
        # rgb_img = torch.stack([img, img, img], dim=-1).cpu().numpy()*255.
        # #print(rgb_img.shape)
        # # print(rgb_img)
        # rgb_img = show_compare(rgb_img, pred_mask, batch.get('anno_mask'),
        #                        batch.get('ignore_mask') if 'ignore_mask' in batch.keys() else None)
        #
        # im = Image.fromarray(np.uint8(rgb_img))
        # im_name = "save_image/jtfn/dsa_image/" + '{}.png'.format(str(img_name[0]))
        # im.save(im_name)

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
