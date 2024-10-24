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
import torch.nn.functional as F

from core.raft_stereo import RAFTStereo
from core.madnet2.madnet2_fusion import MADNet2Fusion

from evaluate_mad_fusion import *
import core.stereo_datasets as datasets

import os

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

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


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def compute_metrics(disp, gt, valid, max_flow=700):
    # add metric (code from RAFT-Stereo)
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == gt.shape, [valid.shape, gt.shape]
    assert not torch.isinf(gt[valid.bool()]).any()

    epe = torch.sum((disp - gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return metrics

def compute_mad_loss(image2, image3, predictions, gt, validgt, max_disp=192):
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(gt ** 2, dim=1).sqrt()

    # exclude extremly large displacements
    validgt = ((validgt >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert validgt.shape == gt.shape, [validgt.shape, gt.shape]
    assert not torch.isinf(gt[validgt.bool()]).any()

    # only use mode 'full++'
    # legacy from original MADNet training (classical average reduction without any weights gives almost identical results)
    loss = [0.001 * F.l1_loss(predictions[0][validgt > 0], gt[validgt > 0], reduction='sum') / 20.,
            0.001 * F.l1_loss(predictions[1][validgt > 0], gt[validgt > 0], reduction='sum') / 20.,
            0.001 * F.l1_loss(predictions[2][validgt > 0], gt[validgt > 0], reduction='sum') / 20.,
            0.001 * F.l1_loss(predictions[3][validgt > 0], gt[validgt > 0], reduction='sum') / 20.,
            0.001 * F.l1_loss(predictions[4][validgt > 0], gt[validgt > 0], reduction='sum') / 20.]
    loss = sum(loss).mean()

    epe = torch.sum((predictions[0] - gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[validgt.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8, betas = (0.9, 0.999)
)

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #         pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150000, gamma=0.5)

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100

    def __init__(self, model, scheduler, name):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=f'runs/{name}')

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

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
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    # model = nn.DataParallel(RAFTStereo(args))
    model = nn.DataParallel(MADNet2Fusion(args))

    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    logger = Logger(model, scheduler, args.name)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.train()
    # remove for madnet2
    # model.module.freeze_bn() # We keep BatchNorm frozen

    validation_frequency = 10000


    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

            # pad images -- code from MADNet 2
            ht, wt = image1.shape[-2], image1.shape[-1]
            pad_ht = (((ht // 128) + 1) * 128 - ht) % 128
            pad_wd = (((wt // 128) + 1) * 128 - wt) % 128
            _pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
            image1 = F.pad(image1, _pad, mode='replicate')
            image2 = F.pad(image2, _pad, mode='replicate')
            guide_proxy = disp_gt.clone()
            guide_proxy = F.pad(guide_proxy, _pad, mode='replicate')

            assert model.training
            # flow_predictions = model(image1, image2, iters=args.train_iters) # for raft-stereo
            pred_disps = model(image1, image2, guide_proxy) # for madnet2 fusion

            assert model.training

            # upsample and remove padding for final prediction -- code from MADNet 2
            pred_disp = F.interpolate(pred_disps[0], scale_factor=4., mode='bilinear')[0] * -20.
            ht, wd = pred_disp.shape[-2:]
            c = [_pad[2], ht - _pad[3], _pad[0], wd - _pad[1]]
            pred_disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]

            # upsample and remove padding from all predictions (if needed for adaptation) -- code from MADNet 2
            pred_disps = [F.interpolate(pred_disps[i], scale_factor=2 ** (i + 2)) * -20. for i in
                          range(len(pred_disps))]

            pred_disps = [pred_disps[i][..., c[0]:c[1], c[2]:c[3]] for i in range(len(pred_disps))]

            image1 = image1[..., c[0]:c[1], c[2]:c[3]]
            image2 = image2[..., c[0]:c[1], c[2]:c[3]]

            # loss, metrics = sequence_loss(flow_predictions, flow, valid) # for ratf-stereo
            loss, metrics = compute_mad_loss(image1, image2, pred_disps, disp_gt, valid)

            # metrics = compute_metrics(pred_disp, disp_gt, valid)

            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                results = validate_things(model.module, iters=args.valid_iters)

                logger.write_dict(results)

                model.train()
                # model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

        if len(train_loader) >= 10000:
            save_path = Path('checkpoints/%d_epoch_%s.pth.gz' % (total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 768], help="size of the random image crops used during training.") # [320, 720] for RAFT-Stereo
    parser.add_argument('--train_iters', type=int, default=16, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

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

    train(args)