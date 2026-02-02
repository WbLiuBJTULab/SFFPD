import torch.nn as nn


class VFETemplate(nn.Module):     # 这个类是一个模板，提供了一个基础框架，具体的功能需要在子类中实现。
    def __init__(self, model_cfg, **kwargs):   # **kwargs：额外的关键字参数，用于传递其他可能需要的配置或参数。
        super().__init__()   # 调用父类的初始化方法
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError
