from __future__ import print_function, division

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.madnet2.madnet2 import MADNet2

from evaluate_mad import *
import core.stereo_datasets as datasets

import torch.nn.functional as F


try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def compute_mad_loss(disp_preds, disp_gt, valid,  max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    loss_weights = [0.08, 0.02, 0.01, 0.005, 0.32]

    for i in range(n_predictions):
        loss = [0.001 * F.l1_loss(disp_preds[0][valid > 0], disp_gt[valid > 0], reduction='sum') / 20.,
                0.001 * F.l1_loss(disp_preds[1][valid > 0], disp_gt[valid > 0], reduction='sum') / 20.,
                0.001 * F.l1_loss(disp_preds[2][valid > 0], disp_gt[valid > 0], reduction='sum') / 20.,
                0.001 * F.l1_loss(disp_preds[3][valid > 0], disp_gt[valid > 0], reduction='sum') / 20.,
                0.001 * F.l1_loss(disp_preds[4][valid > 0], disp_gt[valid > 0], reduction='sum') / 20.]
        accumulated_loss = torch.stack([loss[i] * loss_weights[i] for i in range(len(loss))],0)
        # loss = sum(loss).mean()
        loss = accumulated_loss.mean()

    epe = torch.sum((disp_preds[0] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item() * 100,
        '3px': (epe > 3).float().mean().item()* 100,
        '5px': (epe > 5).float().mean().item()* 100,
    }

    return loss, metrics

# def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
#     """ Loss function defined over sequence of flow predictions """
#
#     n_predictions = len(flow_preds)
#     assert n_predictions >= 1
#     flow_loss = 0.0
#
#     # exlude invalid pixels and extremely large diplacements
#     mag = torch.sum(flow_gt**2, dim=1).sqrt()
#
#     # exclude extremly large displacements
#     valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
#     assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
#     assert not torch.isinf(flow_gt[valid.bool()]).any()
#
#     for i in range(n_predictions):
#         assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
#         # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
#         adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
#         i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
#         i_loss = (flow_preds[i] - flow_gt).abs()
#         assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
#         flow_loss += i_weight * i_loss[valid.bool()].mean()
#
#     epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
#     epe = epe.view(-1)[valid.view(-1)]
#
#     metrics = {
#         'epe': epe.mean().item(),
#         '1px': (epe < 1).float().mean().item(),
#         '3px': (epe < 3).float().mean().item(),
#         '5px': (epe < 5).float().mean().item(),
#     }
#
#     return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=419700, gamma=0.5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=f'runs/{args.name}')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=f'runs/{args.name}')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=f'runs/{args.name}')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(MADNet2(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler)

    if args.restore_ckpt is not None:
        # assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint {args.restore_ckpt}")

    model.cuda()
    model.train()
    # model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10000
    # validation_frequency = 10

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

            assert model.training
            pred_disps = model(image1, image2)
            assert model.training

            # upsample prediction
            pred_disps = [F.interpolate(pred_disps[i], scale_factor=2 ** (i + 2)) * -20. for i in
                          range(len(pred_disps))]

            loss, metrics = compute_mad_loss(pred_disps, disp_gt, valid)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            # if total_steps + 1 % args.decay_step == 0:
            #     scheduler.step()
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path('checkpoints/%s/%d_%s.pth' % (args.name, total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                results = validate_things(model.module, iters=args.valid_iters, log_dir=f'runs/{args.name}')

                logger.write_dict(results)

                model.train()
                # model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path('checkpoints/%s/%d_epoch_%s.pth.gz' % (args.name,total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = 'checkpoints/%s/%s.pth' % (args.name,args.name)
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--warmup', action='store_true', help='use mixed precision')
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 768], help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")
    parser.add_argument('--decay_step', type=int, default=400000, help="decay step for learning rate scheduler")


    # Validation parameters
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)
    Path("checkpoints/%s"%args.name).mkdir(exist_ok=True, parents=True)


    train(args)