import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize
import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
import time
from tensorboardX import SummaryWriter
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size(default: 8)')
    parser.add_argument('--num_workers', default=4, type=int)

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        default='GH_UNet')
    parser.add_argument('--deep_supervision', default=True, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=256, type=int, help='image width(default: 256)')
    parser.add_argument('--input_h', default=256, type=int, help='image height(default: 256)')

    # loss
    parser.add_argument('--loss', default='BCEDiceIoULoss', choices=LOSS_NAMES,
                        help='loss: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceIoULoss)')

    # data
    parser.add_argument('--dataset', default='isic2016',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.jpg', help='image file extension')
    parser.add_argument('--mask_ext', default='._Segmentation.png', help='masks file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR',
                        help='initial learning rate(default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay(default: 1e-4)')
    parser.add_argument('--nesterov', default=True, type=str2bool, help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, dice = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())

    current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    my_writer = SummaryWriter()

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif config['loss'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](num_classes=config['num_classes'],
                                           input_channels=config['input_channels'],
                                           deep_supervision=config['deep_supervision'],
                                           )

    # model.load_from()
    model = model.cuda()
    print("Model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total number of trainable parameters: {trainable_params}")

    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6
    print(f"Total number of parameters in M: {total_params_m:.2f}M")
    print(f"Total number of trainable parameters in M: {trainable_params_m:.2f}M")
    params = filter(lambda p: p.requires_grad, model.parameters())
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    train_img_ids = glob(
        os.path.join('/root/Rolling-Unet/inputs', config['dataset'], 'fold4/train_images', '*' + config['img_ext']))

    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    val_img_ids = glob(os.path.join('inputs', config['dataset'], 'fold4/test_images', '*' + config['img_ext']))

    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    mean = [0.724, 0.619, 0.567]
    std = [0.105, 0.129, 0.151]
    train_transform = A.Compose([
        A.OneOf([
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Resize(config['input_h'], config['input_w'], interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean, std),
            ]),

            A.Compose([
                A.Sharpen(alpha=(0.0, 1.0), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, p=0.5),
                A.Resize(config['input_h'], config['input_w'], interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean, std),
            ]),

            A.Compose([
                A.ElasticTransform(alpha=50, sigma=5, p=0.5),
                A.CropNonEmptyMaskIfExists(height=128, width=128, p=0.5),
                A.RandomGridShuffle(grid=(2, 2), p=0.5),
                A.Resize(config['input_h'], config['input_w'], interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean, std),
            ]),

            A.Compose([
                A.Affine(rotate=(-360, 360), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.4),
                A.CropNonEmptyMaskIfExists(height=128, width=128, p=0.5),
                A.Resize(config['input_h'], config['input_w'], interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean, std),
            ]),

            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(alpha=50, sigma=5, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.4),
                A.Sharpen(alpha=(0.0, 1.0), p=0.5),
                A.Resize(config['input_h'], config['input_w'], interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean, std),
            ]),

            A.Compose([
                A.ElasticTransform(alpha=60, sigma=6, p=0.5),
                A.Affine(rotate=(-180, 180), p=0.4),
                A.RandomGridShuffle(grid=(3, 3), p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit=0.7, p=0.4),
                A.Resize(config['input_h'], config['input_w'], interpolation=cv2.INTER_CUBIC),
                A.Normalize(mean, std),
            ])
        ], p=1.0)
    ])
    val_transform = A.Compose(
        [
            A.Resize(config['input_h'], config['input_w'], interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            transforms.Normalize(),
        ]
    )

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'fold4/train_images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'fold4/train_masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'fold4/test_images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'fold4/test_masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True,
        pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False,
        pin_memory=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        my_writer.add_scalar('loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val_loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val_iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val_dice', val_log['dice'], global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
