from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VirConv8x,VirConvL8x
from .spconv_backbone_CDF_R2P import VirConvAx

__all__ = {
    'VirConv8x': VirConv8x,
    'VirConvL8x': VirConvL8x,
    'VirConvAx': VirConvAx,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
}
