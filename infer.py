import os,sys
import cv2 

import argparse 
import numpy as np

import torch 

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction

from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.custom_data import load_data_with_calib
from mono.utils.do_test import transform_test_data_scalecano

from tqdm import tqdm


def set_argparse():
    parser = argparse.ArgumentParser(description='Infer metric 3d')
    parser.add_argument('--config', required=True, help='train config file path')
    parser.add_argument('--img_dir', required=True, type=str, help='the path of test data')
    parser.add_argument('--out_dir', required=True, type=str, help='the path for output')
    parser.add_argument('--calib_yaml', required=True, type=str, help='the path of calibration yaml')

    # parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument('--load-from', required=True, help='the checkpoint file to load weights from')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', type=int, default=1, help='number of nodes')
    # parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    # parser.add_argument('--launcher', choices=['None', 'pytorch', 'slurm', 'mpi', 'ror'], default='slurm', help='job launcher')
    parser.add_argument('--batch_size', default=1, type=int, help='the batch size for inference')
    parser.add_argument('--norm_out',type=bool,action='store_true' help='output normal')
    args = parser.parse_args()
    return args





if __name__ == "__main__":

    args = set_argparse() 

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.load_from
    cfg.batch_size = args.batch_size


    # load data info
    data_info = {}
    load_data_info('data_info', data_info=data_info)
    cfg.mldb_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)


    # if args.launcher = 'None':
    #     cfg.distributed = False
    # else:
    #     cfg.distributed = True
    #     init_env(args.launcher,cfg)

    test_data = load_data_with_calib(args.img_dir,args.calib_yaml)

    model = get_configured_monodepth_model(cfg,)
    model = torch.nn.DataParallel(model).cuda()

    model, _, _, _ = load_ckpt(cfg.load_from, model ,strict_match=False)
    model.eval()


    os.makedirs(args.out_dir, exist_ok=True)

    normalize_scale = cfg.data_basic.depth_range[1]
    for i in tqdm(range(0,len(test_data),args.batch_size)):
        batch_data = test_data[i:i+args.batch_size]

        img_batch, pad_batch, label_scale_factors, img_origins = [], [], [], [] 

        for an in batch_data:
            img_orig = cv2.imread(an['rgb'])[:,:,::-1].copy()
            img_origins.append(img_orig)

            intrinsic = an['intrinsic']

            rgb, _, pad, label_scale_factor = transform_test_data_scalecano(img_orig, intrinsic, cfg.data_basic)
            
            img_batch.append(rgb)
            pad_batch.append(pad)
            label_scale_factors.append(label_scale_factor)

        data = dict(input=torch.stack(img_batch),cam_model=None)
        pred_depth, confidence, output_dict = model.module.inference(data)


        for j in range(len(batch_data)):
            padj = pad_batch[j]
            img_orig = img_origins[j]
            normj = output_dict['prediction_normal'][j,:]


            # print(pred_depth[j].shape,confidence[j].shape)
            # print(torch.min(confidence[j]),torch.max(confidence[j]))

            depth = pred_depth[j].squeeze()
            depth = depth[padj[0]:depth.shape[0]-padj[1],padj[2]:depth.shape[1]-padj[3]]
            depth = torch.nn.functional.interpolate(depth[None,None,:,:],[img_orig.shape[0],img_orig.shape[1]],mode='bilinear').squeeze()
            depth = depth * normalize_scale/label_scale_factors[j]

            conf = confidence[j].squeeze()
            conf = conf[padj[0]:conf.shape[0]-padj[1],padj[2]:conf.shape[1]-padj[3]]
            conf = torch.nn.functional.interpolate(conf[None,None,:,:],[img_orig.shape[0],img_orig.shape[1]],mode='bilinear').squeeze()

            norm = normj[:3,:,:]
            normH, normW = norm.shape[1:]
            norm = norm[:,padj[0]:normH-padj[1],padj[2]:normW-padj[3]]
            norm = torch.nn.functional.interpolate(norm[None,...],[img_orig.shape[0],img_orig.shape[1]],mode='bilinear').squeeze()
            norm = norm.permute(1,2,0).detach().cpu().numpy()
            norm_norm = np.linalg.norm(norm,axis=-1,keepdims=True)
            norm_norm[norm_norm < 1e-12] = 1e-12
            norm = norm / norm_norm

            depth = (depth > 0) * (depth < 300) *depth 
            depth = depth.detach().cpu().numpy()

            conf = conf.detach().cpu().numpy()

            bname = batch_data[j]['filename']
            bname,ext = os.path.splitext(bname)

            depImg = (np.clip(depth,0,65)*1000.0).astype(np.uint16)
            cv2.imwrite(os.path.join(args.out_dir,bname+'_depth.png'),depImg)

            confImg = (np.clip(conf,0,2)/2*255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(args.out_dir,bname+'_conf.png'),confImg)

            if args.norm_out:
                normImg = (norm+1.0)*0.5*10000.0
                normImg = normImg.astype(np.uint16)
                cv2.imwrite(os.path.join(args.out_dir,bname+'_norm.png'),normImg)

                # normVis = (norm+1.0)*0.5*255.0
                # normVis = normVis.astype(np.uint8)
                # cv2.imwrite(os.path.join(args.out_dir,bname+'_norm_vis.png'),normVis)












