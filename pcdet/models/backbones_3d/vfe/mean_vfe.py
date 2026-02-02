import torch

from .vfe_template import VFETemplate
import numpy as np
 
class MeanVFE(VFETemplate):    # 体素特征编码   通过计算每个体素内所有点的特征的平均值来得到体素的特征表示
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.model = self.model_cfg.get('MODEL',None)

    def get_output_feature_dim(self):
        return self.num_point_features  # 获取最终每个点特征维度  输出的特征维度与输入的特征维度相同，因为每个体素的特征是通过对所有点的特征取平均值得到的。

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C) 体素特征
                voxel_num_points: optional (num_voxels) 每个体素内的点数
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """

        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]   # 区分第几帧
        else:
            rot_num = 1

        for i in range(rot_num):
            if i==0:
                frame_id = ''
            else:
                frame_id = str(i)

            voxel_features, voxel_num_points = batch_dict['voxels'+frame_id], batch_dict['voxel_num_points'+frame_id]
            
            #np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            #with open("voxel_features.txt", "w") as f:
                #f.write("voxel_features:\n")
                #f.write(str(voxel_features.cpu().numpy()))  # 转换为 NumPy 数组以便写入文件

            points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)   #沿着点的维度（dim=1）对每个体素内所有点的特征求总和。  没有点的维度了
            normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features) #加1维度 不小于1
            points_mean = points_mean / normalizer  #每个体素的平均特征 (num_voxels, 1)

            if self.model is not None:
                if self.model == 'max':
                    time_max = voxel_features[:, :, :].max(dim=1, keepdim=False)[0]   #沿着点的维度（dim=1）计算最大值，并取结果的第一个元素（即最大值本身）。
                    points_mean[:, -1] = time_max[:, -1]

            batch_dict['voxel_features'+frame_id] = points_mean.contiguous()  #逐点取得最大值
            #np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            #with open("batch_dict['voxel_features'+frame_id].txt", "w") as f:
                #f.write("batch_dict['voxel_features'+frame_id]:\n")
                #f.write(str(batch_dict['voxel_features'+frame_id].cpu().numpy()))  # 转换为 NumPy 数组以便写入文件

            if 'mm' in batch_dict:
                voxel_features, voxel_num_points = batch_dict['voxels_mm'+frame_id], batch_dict[
                    'voxel_num_points_mm'+frame_id]
                points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
                normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
                points_mean = points_mean / normalizer

                batch_dict['voxel_features_mm'+frame_id] = points_mean.contiguous()  #逐点取得平均值

        return batch_dict
