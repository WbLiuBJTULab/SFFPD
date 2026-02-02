import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
import torch
import cv2
import numpy as np

def get_layer(dim,out_dim,init = None):
    init_func = nn.init.kaiming_normal_
    layers = []
    conv = nn.Conv2d(dim, dim,
                      kernel_size=3, padding=1, bias=True)
    nn.init.normal_(conv.weight, mean=0, std=0.001)
    layers.append(conv)
    layers.append(nn.BatchNorm2d(dim))
    layers.append(nn.ReLU())
    conv2 = nn.Conv2d(dim, out_dim,
                     kernel_size=1, bias=True)

    if init is None:
        nn.init.normal_(conv2.weight, mean=0, std=0.001)
        layers.append(conv2)

    else:
        conv2.bias.data.fill_(init)
        layers.append(conv2)

    return nn.Sequential(*layers)

class AnchorHeadSingle(AnchorHeadTemplate):   # 生成候选区域（anchors）并预测边界框和类别。
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.grid_size = grid_size  # [1408 1600   40]
        self.range = point_cloud_range

        self.voxel_size = (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0]  # 计算体素大小 voxel_size，这是点云范围在某个维度上的距离除以网格大小。都可以


        self.num_anchors_per_location = sum(self.num_anchors_per_location)  # 整个特征图上所有位置的锚点总数。

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )   # 用于预测每个锚点的类别概率
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )


        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_anchor_mask(self,data_dict,shape):  

        stride = np.round(self.voxel_size*8.*10.)  #用于将体素大小转换为特征图中的步长  8是之前3d步长  10是采样

        minx=self.range[0]
        miny=self.range[1]

        points = data_dict["points"]   #点云数据

        mask = torch.zeros(shape[-2],shape[-1])   #用于标记包含点的体素

        mask_large = torch.zeros(shape[-2]//10,shape[-1]//10)   #一个放大10倍的掩码，用于辅助计算

        in_x = (points[:, 1] - minx) / stride   #计算每个点在放大后的分辨率下  索引
        in_y = (points[:, 2] - miny) / stride

        in_x = in_x.long().clamp(max=shape[-1]//10-1)   #限制索引范围  小于 0 的值设置为 0，大于 1 的值设置为 1    -1 是因为索引是从0开始的，所以最大索引应该是网格大小减1
        in_y = in_y.long().clamp(max=shape[-2]//10-1)

 
        mask_large[in_y,in_x] = 1   # 在放大后的掩码中标记包含点云中点的体素   # 标记包含点的体素

        mask_large = mask_large.clone().int().detach().cpu().numpy()   #从计算图中分离出张量，使其不再需要梯度     NumPy 数组只能在 CPU 上创建    

        mask_large_index = np.argwhere( mask_large>0 )  

        mask_large_index = mask_large_index*10 # 由于 mask_large 是在放大10倍的分辨率下计算的，所以需要将索引乘以10来反映这一点。

        index_list=[]

        for i in np.arange(-10, 10, 1):
            for j in np.arange(-10, 10, 1):
                index_list.append(mask_large_index+[i,j])

        index_list = np.concatenate(index_list,0)

        inds = torch.from_numpy(index_list).cuda().long()   # torch.from_numpy 将其转换为一个PyTorch张量。

        mask[inds[:,0],inds[:,1]]=1   # 特征图上 表示这些体素包含点云中的点

        return mask.bool()


    def forward(self, data_dict):

        anchor_mask = self.get_anchor_mask(data_dict,data_dict['st_features_2d'].shape)  # 该掩码用于标记包含点云中点的体素

        new_anchors = []
        for anchors in self.anchors_root:    # 一个包含多个锚点张量的列表，每个张量代表一组预定义的边界框参数（例如，大小、位置和方向）。
            new_anchors.append(anchors[:, anchor_mask, ...])    # 只有那些位于包含点云中点的体素上的锚点才会被用于后续的目标检测步骤。   选择锚点张量中对应于 anchor_mask 为 True 的行

        self.anchors = new_anchors

        st_features_2d = data_dict['st_features_2d']

        cls_preds = self.conv_cls(st_features_2d)
        box_preds = self.conv_box(st_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]   N=批次  只保留那些在掩码中被标记为 True 的预测结果
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds   # 存储各个体素点的预测结果
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(st_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()[:,anchor_mask,:]
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None



        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )   # 为每个锚点分配真实目标（ground truth）。
            self.forward_ret_dict.update(targets_dict)
            data_dict['gt_ious'] = targets_dict['gt_ious']  # ground truth 与预测之间的交并比

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds  # 根据锚点+各体素点的边界框和类别   生成预测的各个边界框和类别
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        if self.model_cfg.get('NMS_CONFIG', None) is not None:
            self.proposal_layer(
                data_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            ) # 调用 proposal_layer 方法应用 NMS 来过滤预测结果

        return data_dict  # ROI（框） 分类值  标签
