import argparse
import os
import time
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import datasets
import models
import utils
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plta
from torchvision import transforms
from SpatialAttention import SpatialAttention
from mmcv.runner import load_checkpoint


def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


# def show_probability(tensor):
#     output = tensor.detach().cpu().numpy()
#     output = output.flatten()
#     print(output.shape)
#     plta.hist(output, bins=20, alph=0.7, color='blue')
#     # plt.xlabel('Probability')
#     # plt.ylabel('Frequency')
#     # plt.show()
#     time.sleep(100000)
#

def tensor2PIL(tensor):
    # print(tensor)
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model,  data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
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

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    val_metric5 = utils.Averager()


    pbar = tqdm(loader, leave=False, desc='val')
    # gmm = GaussianMixture(n_components=2)
    # gmm = BayesianGaussianMixture(n_components=2)

    result1_sum = 0
    a = 0
    with open("numbers.txt", "w") as file:
        for batch in pbar:
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp = batch['inp']
            with torch.no_grad():
                # pred = torch.sigmoid(seg_module(model.infer(inp)))  # 模型的输出
                pred, feat = model.infer(inp)
                pred = torch.sigmoid(pred)
            # pred = torch.add(pred, pred_b)
            # pred = (pred>=0.5).float()


            pre_squeeze = torch.squeeze(pred, dim=0)



            img_save = tensor2PIL(pre_squeeze)  # 转img
            # print(img_save)
            # img_save.show()

            base_path = r"/home/dell/PycharmProjects/crack500_sam/test/image"

            files = sorted(os.listdir(base_path))
                                                                          
            name = files[a]

            name = name[:-4]



            img_save.save(f"/home/dell/PycharmProjects/crack_result/vith_test_best_gmm/{name}.png", "PNG")  # 保存图片

            a = a + 1

            result1, result2, result3, result4, result5 = metric_fn(pred, batch['gt'])

            result1_sum = result1_sum + result1

            val_metric1.add(result1, inp.shape[0])
            val_metric2.add(result2, inp.shape[0])
            val_metric3.add(result3, inp.shape[0])
            val_metric4.add(result4, inp.shape[0])
            val_metric5.add(result5, inp.shape[0])

            #  创建一个列表来记录结果
            result_list = []

            result_list.append('F1: ={:.4f}'.format(result1))
            result_list.append('precision: ={:.4f}'.format(result4))
            result_list.append('recall: ={:.4f}'.format(result5))
            result_list.append('id: ={}'.format(name))

            file.write(str(result_list) + "\n")   # 写入文件中

            if verbose:
                pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
                pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
                pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
                pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))
                pbar.set_description('val {} {:.4f}'.format(metric5, val_metric5.item()))

    print("result_sum为：{}".format(result1_sum/148))
    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_metric5.item()


if __name__ == '__main__':                                                                                                                
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="./configs/cod-sam-vit-h.yaml")
    parser.add_argument('--model', default="/home/dell/PycharmProjects/SAM-Adapter-PyTorch-main/save/gmm/new_4/_cod-sam-vit-h/model_epoch_best.pth")
    # parser.add_argument('--model_b', default="./save/model_vitb_epoch_best_0_120.pth")
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']  #
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)

    model = models.make(config['model']).cuda()
#    model_b = models.make(config['model_b']).cuda()


    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)

    metric1, metric2, metric3, metric4, metric5 = eval_psnr(loader, model,
                                                   data_norm=config.get('data_norm'),
                                                   eval_type=config.get('eval_type'),
                                                   eval_bsize=config.get('eval_bsize'),
                                                   verbose=True)
    print('f1: {:.4f}'.format(metric1))
    print('auc: {:.4f}'.format(metric2))
    print('iou: {:.4f}'.format(metric3))
    print('precision: {:.4f}'.format(metric4))
    print('recall: {:.4f}'.format(metric5))
