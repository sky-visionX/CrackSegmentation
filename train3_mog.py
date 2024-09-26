import argparse
import os
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
from loss_my import bce_dice_loss, gmm_bce_dice_loss, bce_logits_loss
import numpy as np
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F



local_rank = 0
device = torch.device("cuda", 0)


def postprocess_masks(masks):
    masks = F.interpolate(masks, (1024, 1024), mode="bilinear", align_corners=False)
    masks = masks[..., : 1024, : 1024]
    masks = F.interpolate(masks, 1024, mode="bilinear", align_corners=False)
    return masks


def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def eval_psnr(loader, model, eval_type=None):
    model.eval()
    val_loss_list = []
    eval_type = 'f1'
    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4, metric5 = 'f1', 'auc', 'iou', 'precision', 'recall'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    val_metric5 = utils.Averager()


    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            inp = batch['inp']
            pred, feat = model.infer(inp)

            val_loss = bce_dice_loss(pred, batch['gt'])
            val_loss_list.append(val_loss.item())

            pred = torch.sigmoid(pred)

            result1, result2, result3, result4, result5 = metric_fn(pred, batch['gt'])
            val_metric1.add(result1, inp.shape[0])
            val_metric2.add(result2, inp.shape[0])
            val_metric3.add(result3, inp.shape[0])
            val_metric4.add(result4, inp.shape[0])
            val_metric5.add(result5, inp.shape[0])

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()


    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_metric5.item(), metric1, metric2, metric3, metric4, metric5 ,mean(val_loss_list)


def prepare_training():

    if config.get('resume') is not None:
        print("if")
        model = models.make(config['model']).to(device)
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).to(device)
        # all_parameter = list(model.parameters()) + list(automaticWeightedLoss.parameters())
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model,  optimizer, epoch_start, lr_scheduler

def train(train_loader, model, weight, no_rise):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []

    for batch in train_loader:
        for k, v in batch.items():
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']

        pred, feat = model.infer(inp)
        # 高斯
        b, c, w, h = feat.shape
        feat = feat.permute(0, 2, 3, 1)
        feat_flaten = feat.flatten().reshape(-1, c)
        feat_flaten = feat_flaten.cpu().detach().numpy()
        gmm = GaussianMixture(n_components=2, max_iter=500)
        gmm.fit(feat_flaten)

        # 软标签
        pred_gmm = gmm.predict_proba(feat_flaten)

        left_prob = pred_gmm[:, 0]
        right_prob = pred_gmm[:, 1]

        # 生成二分类分割图
        pred_gmm = np.where(right_prob > left_prob, right_prob, 1 - left_prob)
        feat_shape_gmm = (256, 256)
        pred_gmm = np.reshape(pred_gmm, feat_shape_gmm)
        pred_gmm = torch.tensor(pred_gmm, dtype=torch.float32)
        pred_gmm = pred_gmm.to(device)
        # 调整一下这个Pred_GMM
        pred_gmm = pred_gmm.unsqueeze(0).unsqueeze(0)
        pred_gmm = postprocess_masks(pred_gmm)

        count_grater_than_05 = torch.sum(pred_gmm > 0.5).item()
        count_less_than_05 = torch.sum(pred_gmm < 0.5).item()

        if count_grater_than_05 < count_less_than_05:
            pred_gmm = pred_gmm
        else:
            # 像素取反
            pred_gmm = 1 - pred_gmm
        pred_gmm = (pred_gmm > 0.9).float()

        # 只关注标签为目标类别的损失值
        loss_gmm = gmm_bce_dice_loss(pred, pred_gmm)
        loss_gt = bce_dice_loss(pred, gt)
        loss_sum = loss_gt + weight * loss_gmm
        model.optimizer.zero_grad()
        loss_sum.backward()
        model.optimizer.step()

        batch_loss = [torch.zeros_like(loss_sum) for _ in range(dist.get_world_size())]
        dist.all_gather(batch_loss, loss_sum)
        loss_list.extend(batch_loss)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    no_rise = 0
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False
    )
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18
    min_val_loss = 1e8

    weight = 0.02
    weight_mae = 0
    timer = utils.Timer()
    for epoch in range(epoch_start, epoch_max + 1):
        if weight < 0.3:  # 0.3
            weight = weight + 0.03

        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model, weight, no_rise)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0) or (epoch == 1):
            result1, result2, result3, result4, result5, metric1, metric2, metric3, metric4, metric5, val_loss = eval_psnr(val_loader, model,
                                                                                               eval_type=config.get(
                                                                                                   'eval_type'))
            if local_rank == 0:
                log_info.append('val: {}={:.4f}'.format('val_loss', val_loss))
                log_info.append('val: {}={:.4f}'.format(metric1, result1))
                # writer.add_scalars(metric1, {'val': result1}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric2, result2))
                # writer.add_scalars(metric2, {'val': result2}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric3, result3))
                # writer.add_scalars(metric3, {'val': result3}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric4, result4))
                # writer.add_scalars(metric4, {'val': result4}, epoch)
                log_info.append('val: {}={:.4f}'.format(metric5, result5))
                # writer.add_scalars(metric4, {'val': result5}, epoch)
                print(config['eval_type'])

                # F1值
                if result1 > max_val_v:
                    max_val_v = result1
                    save(config, model, save_path, 'best')
                    if no_rise < 3:
                        no_rise = 0
                else:
                    no_rise = no_rise + 1

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save(config, model, save_path, 'loss_best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

                log(', '.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    # print("暂不保存模型")
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':

    import os

    #
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12347'

    torch.distributed.init_process_group('gloo', init_method='env://', rank=0, world_size=1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/cod-sam-vit-h.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=0, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save/gmm/new_6', save_name)

    main(config, save_path, args=args)