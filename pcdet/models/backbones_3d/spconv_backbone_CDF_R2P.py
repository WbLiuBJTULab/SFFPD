from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
import torch.nn as nn
import numpy as np
import torch
from pcdet.datasets.augmentor.X_transform import X_TRANS
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils
from ...ops.pointnet2.pointnet2_stack import voxel_query_utils
from ...utils import voxel_aggregation_utils
from ...utils import common_utils, spconv_utils
from spconv.pytorch import functional as Fsp
import torch.nn.functional as F

class Conv2dNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, norm=None, activation='leaky_relu'):
        super().__init__()

        self.conv_fn = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=norm is None,
        )

        if norm == 'batch_norm':
            self.norm_fn = nn.BatchNorm2d(out_channels)
        elif norm == 'instance_norm':
            self.norm_fn = nn.InstanceNorm2d(out_channels)
        elif norm == 'instance_norm_affine':
            self.norm_fn = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm is None:
            self.norm_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown normalization function: %s' % norm)

        if activation == 'relu':
            self.act_fn = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation is None:
            self.act_fn = nn.Identity()
        else:
            raise NotImplementedError('Unknown activation function: %s' % activation)

    def forward(self, x):
        x = self.conv_fn(x)
        x = self.norm_fn(x)
        x = self.act_fn(x)
        return x
    
def index2points(indices, pts_range=[0, -40, -3, 70.4, 40, 1], voxel_size=[0.05, 0.05, 0.05], stride=8):


    voxel_size = np.array(voxel_size) * stride
    min_x = pts_range[0] + voxel_size[0] / 2
    min_y = pts_range[1] + voxel_size[1] / 2
    min_z = pts_range[2] + voxel_size[2] / 2

    new_indices = indices.clone().float()
    indices_float = indices.clone().float()
    new_indices[:, 1] = indices_float[:, 3] * voxel_size[0] + min_x
    new_indices[:, 2] = indices_float[:, 2] * voxel_size[1] + min_y
    new_indices[:, 3] = indices_float[:, 1] * voxel_size[2] + min_z

    return new_indices


def index2uv3d(indices, batch_size, calib, stride, x_trans_train, trans_param):

    new_uv = indices.clone().int()
    for b_i in range(batch_size):
        cur_in = indices[indices[:, 0] == b_i]
        cur_pts = index2points(cur_in, stride=stride)
        if trans_param is not None:
            transed = x_trans_train.backward_with_param({'points': cur_pts[:, 1:4],
                                                         'transform_param': trans_param[b_i]})
            cur_pts = transed['points'].cpu().numpy()
        else:
            cur_pts = cur_pts[:, 1:4].cpu().numpy()

        pts_rect = calib[b_i].lidar_to_rect(cur_pts[:, 0:3])
        pts_img, pts_rect_depth = calib[b_i].rect_to_img(pts_rect)
        pts_img = pts_img.astype(np.int32)
        pts_img = torch.from_numpy(pts_img).to(new_uv.device)
        new_uv[indices[:, 0] == b_i, 2:4] = pts_img

    new_uv[:, 1] = 1
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max=1400) // stride
    new_uv[:, 3] = torch.clamp(new_uv[:, 3], min=0, max=600) // stride
    return new_uv


def index2uv(indices, batch_size, calib, stride, x_trans_train, trans_param):


    new_uv = indices.new(size=(indices.shape[0], 3))
    depth = indices.new(size=(indices.shape[0], 1)).float()
    for b_i in range(batch_size):
        cur_in = indices[indices[:, 0] == b_i]
        cur_pts = index2points(cur_in, stride=stride) #
        if trans_param is not None:
            transed = x_trans_train.backward_with_param({'points': cur_pts[:, 1:4],
                                                         'transform_param': trans_param[b_i]})
            cur_pts = transed['points']  
        else:
            cur_pts = cur_pts[:, 1:4] 

        pts_rect = calib[b_i].lidar_to_rect_cuda(cur_pts[:, 0:3])
        pts_img, pts_rect_depth = calib[b_i].rect_to_img_cuda(pts_rect)

        pts_img = pts_img.int()

        new_uv[indices[:, 0] == b_i, 1:3] = pts_img

        depth[indices[:, 0] == b_i, 0] = pts_rect_depth[:]
    new_uv[:, 0] = indices[:, 0]
    new_uv[:, 1] = torch.clamp(new_uv[:, 1], min=0, max=1400 - 1) // stride
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max=600 - 1) // stride

    return new_uv, depth


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m


def post_act_block2d(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                     conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )
    return m
def layer_voxel_discard(sparse_t, rat=0.15):
    if rat == 0:
        return
    len = sparse_t.features.shape[0]
    randoms = np.random.permutation(len)
    randoms = torch.from_numpy(randoms[0:int(len * (1 - rat))]).to(sparse_t.features.device)
    sparse_t = replace_feature(sparse_t, sparse_t.features[randoms])
    sparse_t.indices = sparse_t.indices[randoms]


class NRConvBlock(nn.Module):


    def __init__(self, input_c=16, output_c=16, stride=1, padding=1, indice_key='vir1', conv_depth=False):
        super(NRConvBlock, self).__init__()
        self.stride = stride
        block = post_act_block
        block2d = post_act_block2d
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_depth = conv_depth

        if self.stride > 1:
            self.down_layer = block(input_c,
                                    output_c,
                                    3,
                                    norm_fn=norm_fn,
                                    stride=stride,
                                    padding=padding,
                                    indice_key=('sp' + indice_key),
                                    conv_type='spconv')
        c1 = input_c

        if self.stride > 1:
            c1 = output_c
        if self.conv_depth:
            c1 += 4

        c2 = output_c

        self.d3_conv1 = block(c1,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm1' + indice_key))
        self.d2_conv1 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm3' + indice_key))

        self.d3_conv2 = block(c2 // 2,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm2' + indice_key))
        self.d2_conv2 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm4' + indice_key))
        
        self.mlp1 = Conv2dNormRelu(c2 // 2, 2, norm=None, activation="sigmoid")
        self.mlp2 = Conv2dNormRelu(c2 // 2, 2, norm=None, activation="sigmoid")
        self.weight_fc1 = nn.Linear(c2 // 2, c2 // 2)
        self.weight_fc2 = nn.Linear(c2 // 2, c2 // 2)

        nn.init.kaiming_normal_(self.weight_fc1.weight, mode='fan_out', nonlinearity='sigmoid')
        nn.init.kaiming_normal_(self.weight_fc2.weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, sp_tensor, batch_size, calib, stride, x_trans_train, trans_param):
        if self.stride > 1:
            sp_tensor = self.down_layer(sp_tensor)
        d3_feat1 = self.d3_conv1(sp_tensor)
        d3_feat2 = self.d3_conv2(d3_feat1)
        uv_coords, depth = index2uv(d3_feat2.indices, batch_size, calib, stride, x_trans_train, trans_param)
        d2_sp_tensor1 = spconv.SparseConvTensor(
            features=d3_feat2.features,
            indices=uv_coords.int(),
            spatial_shape=[1600, 600],
            batch_size=batch_size
        )
        d2_feat1 = self.d2_conv1(d2_sp_tensor1)
        d2_feat2 = self.d2_conv2(d2_feat1)
        d3_feat3 = replace_feature(d3_feat2, torch.cat([d3_feat2.features, d2_feat2.features], -1))      
        return d3_feat3


class VirConvAx(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size,  **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.x_trans_train = X_TRANS()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features = model_cfg.OUT_FEATURES
        self.layer_discard_rate = model_cfg.LAYER_DISCARD_RATE

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                  conv_type='spconv'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        self.conv_out_mm = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2_fusuion'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        
        if self.model_cfg.get('MM', False):
            self.vir_conv1 = NRConvBlock(input_channels, num_filters[0], stride=1, indice_key='vir1')
            self.vir_conv2 = NRConvBlock(num_filters[0], num_filters[1], stride=2, indice_key='vir2')
            self.vir_conv3 = NRConvBlock(num_filters[1], num_filters[2], stride=2, indice_key='vir3')
            self.vir_conv4 = NRConvBlock(num_filters[2], num_filters[3], stride=2, padding=(0, 1, 1),
                                          indice_key='vir4')

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features
        self.downscale_blocks = spconv.SparseSequential(
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                  conv_type='spconv'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        self.point_cloud_range = [0, -40, -3, 70.4, 40, 1]
        self.voxel_size = [0.05, 0.05, 0.05]
        self.relu = nn.ReLU()
        self.attention = Attention1(channels = [64, 64])

    def get_voxel_center_features(self,batch_dict):
        centroid_coords = {}
        centroid_features = {}
        centroids_voxel_idxs_list = {}
        centroids_voxel_idxs_features_list = {}
        centroids_all, centroid_voxel_idxs_all = voxel_aggregation_utils.get_centroids_per_voxel_layer(
            batch_dict['points'],
            ['x_conv3', 'x_conv4'],
            batch_dict['multi_scale_3d_strides'],
            [0.05, 0.05, 0.05],
            [0, -40, -3, 70.4, 40, 1]) 

        for feature_location in ['x_conv3', 'x_conv4']:        
            centroids = centroids_all[feature_location][:, :4]
            centroids_voxel_idxs = centroid_voxel_idxs_all[feature_location]
            x_conv = batch_dict['multi_scale_3d_features'][feature_location]  
            overlapping_voxel_feature_indices_nonempty, overlapping_voxel_feature_nonempty_mask = \
                voxel_aggregation_utils.get_nonempty_voxel_feature_indices(centroids_voxel_idxs, x_conv)
            x_conv_features = torch.zeros((centroids.shape[0], x_conv.features.shape[-1]), dtype=x_conv.features.dtype,device=centroids.device)
            x_conv_features_fake = x_conv.features   
            x_conv_features[overlapping_voxel_feature_nonempty_mask] = x_conv_features_fake[overlapping_voxel_feature_indices_nonempty]
            centroid_features[feature_location] = x_conv_features[overlapping_voxel_feature_nonempty_mask] 
            centroid_coords[feature_location] = centroids[overlapping_voxel_feature_nonempty_mask]  
            x_conv_indices_features = torch.zeros((centroids_voxel_idxs.shape[0], x_conv.features.shape[-1]), dtype=x_conv.features.dtype,device=centroids.device)
            x_conv_features_fake = x_conv.features
            x_conv_indices_features[overlapping_voxel_feature_nonempty_mask] = x_conv_features_fake[overlapping_voxel_feature_indices_nonempty]
            centroids_voxel_idxs_features_list[feature_location] = x_conv_indices_features[overlapping_voxel_feature_nonempty_mask]
            centroids_voxel_idxs_list[feature_location] = centroids_voxel_idxs[overlapping_voxel_feature_nonempty_mask]
        return centroid_coords, centroid_features, centroids_voxel_idxs_list, centroids_voxel_idxs_features_list
    def gate_pesudo_voxel(self, batch_dict,  centorid_voxel,centorid_coords, centroid_features, batch_size):
        stage_out = []
        pesudo_features = {}
        QUERY_RANGE = {'x_conv3': [6, 6, 6], 'x_conv4': [8, 8, 8]}  
        POOL_RADIUS = {'x_conv3': 2, 'x_conv4': 4}   
        NSAMPLE = {'x_conv3': 20, 'x_conv4': 20}   
        dist_thresh = {'x_conv3': 10, 'x_conv4': 6}  
        cross_gating_in_list = []
        spatial_shape = {'x_conv3': [21, 400, 352], 'x_conv4': [10, 200, 176]}
        for x_onvx in ['x_conv3','x_conv4']:
            voxel_only_2D_indices_tmp = centorid_coords[x_onvx]
            voxel_only_2D_voxel_tmp =  centorid_voxel[x_onvx]
            voxel_only_2D_features = centroid_features[x_onvx]
            voxel_3D_indices = batch_dict['multi_scale_3d_features_mm'][x_onvx].indices            
            voxel_3D_Downtimes = batch_dict['multi_scale_3d_strides'][x_onvx]
            voxel_3D_F = batch_dict['multi_scale_3d_features_mm'][x_onvx].features
            v2p_ind_tensor = spconv_utils.generate_voxel2pinds(batch_dict['multi_scale_3d_features_mm'][x_onvx])
            _, grouped_features = self.Center_voxel_Query_NN_fast(voxel_only_2D_indices_tmp,voxel_only_2D_voxel_tmp, voxel_3D_indices, POOL_RADIUS[x_onvx],
                                                                QUERY_RANGE[x_onvx], NSAMPLE[x_onvx], dist_thresh[x_onvx], voxel_3D_Downtimes, v2p_ind_tensor, voxel_3D_F)
            voxel_only_2D_features = self.attention(grouped_features, voxel_only_2D_features)
            pesudo_conv = spconv.SparseConvTensor(
                features=voxel_only_2D_features,
                indices=voxel_only_2D_voxel_tmp.int(),
                spatial_shape=spatial_shape[x_onvx], 
                batch_size=batch_size
            )
            pesudo_features[x_onvx] = pesudo_conv
        for stage_id in (3,4):
            if stage_id == 3:
                voxel_stage_out_ds = self.downscale_blocks(pesudo_features[f'x_conv{stage_id}'])
                stage_out.append(voxel_stage_out_ds)
            else:
                voxel_stage_out = Fsp.sparse_add(pesudo_features[f'x_conv{stage_id}'], stage_out[0])
                stage_out.append(voxel_stage_out)
        return stage_out

    def Center_voxel_Query_NN_fast(self, query_coords, query_voxels, key, radius, query_range, max_cluster_samples, dist_thresh, Downtimes, v2p_ind_tensor, features):
        xyz_coords = query_coords.unsqueeze(0)[:,:,[1,2,3]] # [xyz]
        xyz_batch_cnt = xyz_coords.new_zeros(1).int().fill_(xyz_coords.shape[1])
        new_xyz= key
        new_xyz_center = common_utils.get_voxel_centers( #xyz
            new_xyz[:,1:4],
            downsample_times=Downtimes,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )  

        cur_voxel_xyz_batch_cnt = new_xyz.new_zeros(1).int().fill_(new_xyz.shape[0])  
        query_group_idx, empty_ball_mask = voxel_query_utils.voxel_query(query_range, radius, max_cluster_samples, new_xyz_center.contiguous(), xyz_coords.contiguous().view(-1,3),
                                                                 query_voxels.int().contiguous().view(-1,4),v2p_ind_tensor) 
        query_group_idx[empty_ball_mask] = 0 
        grouped_features = pointnet2_utils.grouping_operation(features,cur_voxel_xyz_batch_cnt, query_group_idx, xyz_batch_cnt) 
        grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0) 
        grouped_features = self.relu(grouped_features)
        grouped_features = F.max_pool2d(
                    grouped_features, kernel_size=[1, grouped_features.size(3)]
                ).squeeze(dim=-1) 
        grouped_features = grouped_features.squeeze(dim=0).permute(1, 0)
        grouped_features[empty_ball_mask] = 0  
        return query_group_idx, grouped_features
        
    def decompose_tensor(self, tensor, i, batch_size):
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        batch_size = batch_dict['batch_size']
        all_lidar_feat = []
        all_lidar_coords = []

        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * 4]

        if self.training:
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features' + rot_num_id], batch_dict[
                    'voxel_coords' + rot_num_id]

                batch_size = batch_dict['batch_size']
                input_sp_tensor = spconv.SparseConvTensor(
                    features=voxel_features,
                    indices=voxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                x = self.conv_input(input_sp_tensor)

                x_conv1 = self.conv1(x)
                x_conv2 = self.conv2(x_conv1)
                x_conv3 = self.conv3(x_conv2)
                x_conv4 = self.conv4(x_conv3)

                out = self.conv_out(x_conv4)

                batch_dict.update({
                    'encoded_spconv_tensor' + rot_num_id: out,
                    'encoded_spconv_tensor_stride' + rot_num_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features' + rot_num_id: {
                        'x_conv1': x_conv1,
                        'x_conv2': x_conv2,
                        'x_conv3': x_conv3,
                        'x_conv4': x_conv4,
                    },
                    'multi_scale_3d_strides' + rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })
        else:
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features' + rot_num_id], batch_dict[
                    'voxel_coords' + rot_num_id]
                all_lidar_feat.append(voxel_features)
                new_coord = voxel_coords.clone()
                new_coord[:, 3] += i * self.sparse_shape[2]
                all_lidar_coords.append(new_coord)

            all_lidar_feat = torch.cat(all_lidar_feat, 0)
            all_lidar_coords = torch.cat(all_lidar_coords)

            input_sp_tensor = spconv.SparseConvTensor(
                features=all_lidar_feat,
                indices=all_lidar_coords.int(),
                spatial_shape=new_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)
            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            out = self.conv_out(x_conv4)

            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
                this_out = self.decompose_tensor(out, i, batch_size)

                batch_dict.update({
                    'encoded_spconv_tensor' + rot_num_id: this_out,
                    'encoded_spconv_tensor_stride' + rot_num_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features' + rot_num_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides' + rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        for i in range(rot_num):
            if i == 0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            if self.model_cfg.get('MM', False):
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm' + rot_num_id], batch_dict[
                    'voxel_coords_mm' + rot_num_id]

                newinput_sp_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=newvoxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                if self.training:
                    layer_voxel_discard(newinput_sp_tensor, self.layer_discard_rate)

                calib = batch_dict['calib']
                batch_size = batch_dict['batch_size']
                if 'aug_param' in batch_dict:
                    trans_param = batch_dict['aug_param']
                else:
                    trans_param = None
                if 'transform_param' in batch_dict:
                    trans_param = batch_dict['transform_param'][:, i, :]

                newx_conv1 = self.vir_conv1(newinput_sp_tensor, batch_size, calib, 1, self.x_trans_train, trans_param)

                if self.training:
                    layer_voxel_discard(newx_conv1, self.layer_discard_rate)

                newx_conv2 = self.vir_conv2(newx_conv1, batch_size, calib, 2, self.x_trans_train, trans_param)

                if self.training:
                    layer_voxel_discard(newx_conv2, self.layer_discard_rate)

                newx_conv3 = self.vir_conv3(newx_conv2, batch_size, calib, 4, self.x_trans_train, trans_param)

                if self.training:
                    layer_voxel_discard(newx_conv3, self.layer_discard_rate)

                newx_conv4 = self.vir_conv4(newx_conv3, batch_size, calib, 8, self.x_trans_train, trans_param)

                batch_dict.update({
                    'encoded_spconv_tensor_stride_mm' + rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm' + rot_num_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                        'x_conv3': newx_conv3,
                        'x_conv4': newx_conv4,
                    },
                    'multi_scale_3d_strides' + rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })
        centroid_coords, centroid_features, centroids_voxel_idxs_list, centroids_voxel_idxs_features_list=self.get_voxel_center_features(batch_dict)
        gate_pesudo_voxel = self.gate_pesudo_voxel(batch_dict, centroids_voxel_idxs_list, centroid_coords, centroid_features, batch_size)
        fusion_out = self.conv_out_mm(gate_pesudo_voxel[-1])
        batch_dict.update({
            'encoded_spconv_tensor_mm':fusion_out,
            'encoded_spconv_tensor_stride_mm':8
        })
        return batch_dict

class Attention1(nn.Module):
    def __init__(self, channels):
        super(Attention1, self).__init__()
        self.pseudo_in, self.valid_in = channels
        middle = self.valid_in // 4
        self.fc1 = nn.Linear(self.pseudo_in, middle)
        self.fc2 = nn.Linear(self.valid_in, middle)
        self.fc3 = nn.Linear(2*middle, 2)
        self.fc4 = nn.Linear(2*self.pseudo_in, self.pseudo_in)
        self.conv1 = nn.Sequential(nn.Linear(self.pseudo_in, self.valid_in),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(self.valid_in, self.valid_in),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())
        self.bn1 = torch.nn.BatchNorm1d(self.valid_in)

    def forward(self, pseudo_feas, valid_feas):
        pseudo_feas_f_ = self.fc1(pseudo_feas) # ([8868, 16])
        valid_feas_f_ = self.fc2(valid_feas) # ([8868, 16])
        pseudo_valid_feas_f = torch.cat([pseudo_feas_f_, valid_feas_f_],dim=-1)
        weight = torch.sigmoid(self.fc3(pseudo_valid_feas_f))
        pseudo_weight = weight[:,0].view(-1, 1)# ([8868, 1])
        valid_weight = weight[:,1].view(-1, 1)
        pseudo_feas = self.conv1(pseudo_feas)
        pseudo_features_att = self.conv1(pseudo_feas)* pseudo_weight
        valid_features_att     =  self.conv2(valid_feas)   *  valid_weight
        fusion_features = torch.cat([valid_features_att, pseudo_features_att], dim=1) 
        fusion_features = F.relu(self.bn1(self.fc4(fusion_features)))   
        return fusion_features

