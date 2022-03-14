import argparse
import glob
from pathlib import Path

# try:
#     import open3d
#     from visual_utils import open3d_vis_utils as V
#     OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#     OPEN3D_FLAG = False

import os
import numpy as np
import torch
import torch.onnx
import pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import WatoDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from JustSECOND.src.second_net_iou import SECONDNetIoU
from JustSECOND.src.dataset import PointCloudPrep


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

from time import perf_counter
def timer(f, data_dict):   
    start = perf_counter()
    f(data_dict)
    return (1000 * (perf_counter() - start))

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = WatoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )
    pc_prepper = PointCloudPrep()
    root_path = "/home/road/combined_cars/"
    lidar_paths = glob.glob(root_path + "pc*.npy")

    # model_original = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model = SECONDNetIoU(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES)) #pc_prepper
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        data_dict = next(iter(demo_dataset))
        data_dict = demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        # logger.info(data_dict)
        
        # a = model(data_dict)
        # logger.info(np.mean([timer(model_original, data_dict) for _ in range(100)]))


        del data_dict
        points = np.load(root_path + "pc01651.npy")
        data_dict = pc_prepper.prepare_data(points)

        print(data_dict.keys())
        print("points", data_dict["points"].size())
        print("voxels", data_dict["voxels"].size())
        print("voxel_coords", data_dict["voxel_coords"].size())
        print("voxel_num_points", data_dict["voxel_num_points"].size())

        # a = torch.jit.script(model, (data_dict))
        
        a = model(data_dict)
                    

    logger.info('Demo done.')


if __name__ == '__main__':
    main()