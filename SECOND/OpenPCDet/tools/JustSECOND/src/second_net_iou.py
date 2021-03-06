from typing import Dict
import os
import numpy as np
import torch
import torch.nn as nn

from .utils import class_agnostic_nms, logger, find_all_spconv_keys, BATCH_SIZE
from .mean_vfe import MeanVFE
from .voxel_backbone import VoxelBackBone8x
from .height_compression import HeightCompression
from .base_bev_backbone import BaseBEVBackbone
from .anchor_head_single import AnchorHeadSingle
from .second_head import SECONDHead
from .dataset import NUM_POINT_FEATURES, GRID_SIZE, POINT_CLOUD_RANGE, VOXEL_SIZE

# BASE_DIR = "/home/OpenPCDet/tools/JustSECOND/"

class SECONDNetIoU(nn.Module):
    def __init__(self, model_cfg, num_class=3):
        super().__init__()

        self.num_class = num_class
        self.class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 
            'backbone_2d', 'dense_head', 'roi_head'
        ]
        self.model_cfg = model_cfg
        self.module_list = self.build_networks()

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': NUM_POINT_FEATURES,
            'num_point_features': NUM_POINT_FEATURES,
            'grid_size': GRID_SIZE,
            'point_cloud_range': POINT_CLOUD_RANGE,
            'voxel_size': VOXEL_SIZE,
            'depth_downsample_factor': None
        }
        # logger.info("model_info_dict {}".format(model_info_dict))
        # model_info_dict = {
        #     'module_list': [],
        #     'num_rawpoint_features': 3, # ["x", "y", "z"]
        #     'num_point_features': 3, # ["x", "y", "z"]
        #     'grid_size': np.array([1504, 400, 40], dtype=np.int64), # pc max-min range / voxel_size = [150.4, 150.4, 6] OR [100, 40, 6] / [0.1, 0.1, 0.15] = [1000, 400, 40]
        #     'point_cloud_range': np.array([-50, -20, -2, 50, 20, 4], dtype=np.float32),
        #     'voxel_size': [0.1, 0.1, 0.15],
        #     'depth_downsample_factor': None
        # }
        
        vfe_module = MeanVFE(
            num_point_features=model_info_dict['num_rawpoint_features'])
        model_info_dict["module_list"].append(vfe_module)
        self.add_module("vfe", vfe_module)

        backbone_3d_module = VoxelBackBone8x(
            input_channels=model_info_dict["num_point_features"], 
            grid_size=model_info_dict["grid_size"])
        model_info_dict["module_list"].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels
        self.add_module("backbone_3d", backbone_3d_module)

        map_to_bev_module = HeightCompression(
            num_bev_features=self.model_cfg.MAP_TO_BEV.NUM_BEV_FEATURES
        )
        # map_to_bev_module = torch.jit.load(BASE_DIR + "map_to_bev_module_01.pt")
        model_info_dict["module_list"].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        self.add_module("map_to_bev_module", map_to_bev_module)

        backbone_2d_module = BaseBEVBackbone(
            model_cfg=self.model_cfg.BACKBONE_2D, 
            input_channels=model_info_dict['num_bev_features']
        )
        # backbone_2d_module = torch.jit.load(BASE_DIR + "backbone_2d_01.pt")
        model_info_dict["module_list"].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        self.add_module("backbone_2d", backbone_2d_module)

        dense_head_module = AnchorHeadSingle(
            model_cfg=self.model_cfg.DENSE_HEAD, 
            input_channels=model_info_dict['num_bev_features'], 
            num_class=self.num_class, 
            class_names=self.class_names, 
            grid_size=model_info_dict['grid_size'], 
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=True
        )
        # dense_head_module = torch.jit.load(BASE_DIR + "dense_head_01.pt")
        model_info_dict["module_list"].append(dense_head_module)
        self.add_module("dense_head", dense_head_module)

        point_head_module = SECONDHead(
            input_channels=model_info_dict['num_point_features'], 
            model_cfg=self.model_cfg.ROI_HEAD, 
            num_class=self.num_class
        )
        model_info_dict["module_list"].append(point_head_module)
        self.add_module("roi_head", point_head_module)

        return model_info_dict['module_list']

    def forward(self, batch_dict: Dict[str, torch.Tensor]):
        # batch_dict['dataset_cfg'] = self.dataset

        batch_dict = self.module_list[0](batch_dict)
        batch_dict = self.module_list[1](batch_dict)
        
        # a2 = torch.jit.script(self.module_list[2])
        # a3 = torch.jit.script(self.module_list[3])
        # a4 = torch.jit.script(self.module_list[4])
        # a5 = torch.jit.script(self.module_list[5])
        
        # torch.jit.save(a2, BASE_DIR + "map_to_bev_module_01.pt")
        # torch.jit.save(a3, BASE_DIR + "backbone_2d_01.pt")
        # torch.jit.save(a4, BASE_DIR + "dense_head_01.pt")
        # torch.jit.save(a5, BASE_DIR + "roi_head_01.pt")
        

        str_modules = ["map_to_bev", "backbone_2d", "dense_head", "roi_head"]
        for index, cur_module in enumerate(self.module_list[2:]):
            # a = torch.jit.trace(cur_module, (batch_dict))
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    @staticmethod
    def cal_scores_by_npoints(cls_scores, iou_scores, num_points_in_gt, cls_thresh=10, iou_thresh=100):
        """
        Args:
            cls_scores: (N)
            iou_scores: (N)
            num_points_in_gt: (N, 7+c)
            cls_thresh: scalar
            iou_thresh: scalar
        """
        assert iou_thresh >= cls_thresh
        alpha = torch.zeros(cls_scores.shape, dtype=torch.float32).cuda()
        alpha[num_points_in_gt <= cls_thresh] = 0
        alpha[num_points_in_gt >= iou_thresh] = 1
        
        mask = ((num_points_in_gt > cls_thresh) & (num_points_in_gt < iou_thresh))
        alpha[mask] = (num_points_in_gt[mask] - 10) / (iou_thresh - cls_thresh)
        
        scores = (1 - alpha) * cls_scores + alpha * iou_scores

        return scores

    def set_nms_score_by_class(self, iou_preds, cls_preds, label_preds, score_by_class):
        n_classes = torch.unique(label_preds).shape[0]
        nms_scores = torch.zeros(iou_preds.shape, dtype=torch.float32).cuda()
        for i in range(n_classes):
            mask = label_preds == (i + 1)
            class_name = self.class_names[i]
            score_type = score_by_class[class_name]
            if score_type == 'iou':
                nms_scores[mask] = iou_preds[mask]
            elif score_type == 'cls':
                nms_scores[mask] = cls_preds[mask]
            else:
                raise NotImplementedError

        return nms_scores

    def update_global_step(self):
        self.global_step += 1
        
    def post_processing(self, batch_dict: Dict[str, torch.Tensor]):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size'] # BATCH_SIZE
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            iou_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = batch_dict['roi_scores'][batch_mask]

            src_iou_preds = iou_preds
            src_box_preds = box_preds
            src_cls_preds = cls_preds
            assert iou_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                iou_preds = torch.sigmoid(iou_preds)
                cls_preds = torch.sigmoid(cls_preds)

            iou_preds, label_preds = torch.max(iou_preds, dim=-1)
            label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels', False) else label_preds + 1
            
            nms_scores = iou_preds
        
            selected, selected_scores = class_agnostic_nms(
                box_scores=nms_scores, box_preds=box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
                score_thresh=post_process_cfg.SCORE_THRESH
            )

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'pred_cls_scores': cls_preds[selected],
                'pred_iou_scores': iou_preds[selected]
            }

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch

    def load_params_from_file(self, filename, to_cpu=False, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])
        logger.info('==> Done')

        return it, epoch


    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state