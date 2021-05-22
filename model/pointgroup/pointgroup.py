import torch
import torch.nn as nn
import spconv
from spconv.modules import SparseModule
import functools
from collections import OrderedDict
import sys
sys.path.append('../../')
from torch.nn import functional as F
from lib.pointgroup_ops.functions import pointgroup_ops
from util import utils
import numpy as np
from model.transformer import TransformerEncoder


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output.features += self.i_branch(identity).features

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, use_backbone_transformer=False, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes
        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)
        if len(nPlanes)<=2 and use_backbone_transformer:
            d_model = 128
            self.before_transformer_linear = nn.Linear(nPlanes[0], d_model)
            self.transformer = TransformerEncoder(d_model=d_model, N=2, heads=4, d_ff=64)
            self.after_transformer_linear = nn.Linear(d_model, nPlanes[0])
        else:
            self.before_transformer_linear = None
            self.transformer = None
            self.after_transformer_linear = None
        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, use_backbone_transformer, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)


    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        if self.before_transformer_linear:
            batch_ids = output.indices[:, 0]
            xyz = output.indices[:, 1:].float()
            feats = output.features
            before_params_feats = self.before_transformer_linear(feats)
            feats = self.transformer(xyz=xyz, features=before_params_feats, batch_ids=batch_ids)
            feats = self.after_transformer_linear(feats)
            output.features = feats

        return output

from util.warpper import Conv1d, BatchNorm1d
def conv_with_kaiming_uniform(norm=None, activation=None, use_sep=False):
    def make_conv(in_channels, out_channels):
        conv_func = Conv1d
        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1

        conv = conv_func(in_channels,
                         out_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         groups=groups,
                         bias=(norm is None))

        nn.init.kaiming_uniform_(conv.weight, a=1)
        if norm is None:
            nn.init.constant_(conv.bias, 0)

        module = [conv,]
        if norm is not None and len(norm) > 0:
            norm_module = BatchNorm1d(out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv

class PointGroup(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        input_c = cfg.input_channel
        m = cfg.m
        classes = cfg.classes
        block_reps = cfg.block_reps
        block_residual = cfg.block_residual

        self.cluster_radius = cfg.cluster_radius
        self.cluster_meanActive = cfg.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster_npoint_thre

        self.score_scale = cfg.score_scale
        self.score_fullscale = cfg.score_fullscale
        self.mode = cfg.score_mode

        self.prepare_epochs = cfg.prepare_epochs

        # self.pretrain_path = cfg.pretrain_path
        # self.pretrain_module = cfg.pretrain_module
        self.fix_module = cfg.fix_module

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock

        if cfg.use_coords:
            input_c += 3

        #### backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )




        self.unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, use_backbone_transformer=cfg.use_backbone_transformer, indice_key_id=1)

        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        #### semantic segmentation
        # self.linear = nn.Linear(m, classes) # bias(default): True
        self.linear = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU(),
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU(),
            nn.Linear(m, classes, bias=True),
        )


        ################################
        ################################
        ################################
        ### for instance embedding
        self.output_dim = 16
        self.mask_conv_num = 3
        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        mask_tower = []
        for i in range(self.mask_conv_num):
            mask_tower.append(conv_block(m, m))
        mask_tower.append(nn.Conv1d(
            m,  self.output_dim, 1
        ))
        self.add_module('mask_tower', nn.Sequential(*mask_tower))

        ### convolution before the condinst take place (convolution num before the generated parameters take place)
        before_embedding_conv_num = 1
        conv_block = conv_with_kaiming_uniform("BN", activation=True)
        before_embedding_tower = []
        for i in range(before_embedding_conv_num-1):
            before_embedding_tower.append(conv_block(m, m))
        before_embedding_tower.append(conv_block(m, self.output_dim))
        self.add_module("before_embedding_tower", nn.Sequential(*before_embedding_tower))

        ### cond inst generate parameters for
        USE_COORDS = True
        self.use_coords = USE_COORDS
        self.embedding_conv_num = 2
        weight_nums = []
        bias_nums = []
        for i in range(self.embedding_conv_num):
            if i ==0:
                if USE_COORDS:
                    weight_nums.append((self.output_dim+3) * self.output_dim)
                else:
                    weight_nums.append(self.output_dim * self.output_dim)
                bias_nums.append(self.output_dim)
            elif i == self.embedding_conv_num-1:
                weight_nums.append(self.output_dim)
                bias_nums.append(1)
            else:
                weight_nums.append(self.output_dim*self.output_dim)
                bias_nums.append(self.output_dim)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.controller = nn.Conv1d(self.output_dim, self.num_gen_params, kernel_size=1)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)



        #### offset
        self.offset = nn.Sequential(
            nn.Linear(m, m, bias=True),
            norm_fn(m),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(m, 3, bias=True)

        #### score branch
        self.score_unet = UBlock([m, 2*m], norm_fn, 2, block, indice_key_id=1)
        self.score_outputlayer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )



        self.apply(self.set_bn_init)

        self.threshold_ins = cfg.threshold_ins
        self.min_pts_num = cfg.min_pts_num
        #### fix parameter
        module_map = {'input_conv': self.input_conv, 'unet': self.unet, 'output_layer': self.output_layer,
                      'linear': self.linear, 'offset': self.offset, 'offset_linear': self.offset_linear,
                      'score_unet': self.score_unet, 'score_outputlayer': self.score_outputlayer}

        for m in self.fix_module:
            mod = module_map[m]
            for param in mod.parameters():
                param.requires_grad = False


    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        '''
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        '''
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean0 = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean = torch.index_select(clusters_coords_mean0, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        out_coords, inp_map, out_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # output_coords: M * (1 + 3) long
        # input_map: sumNPoint int
        # output_map: M * (maxActive + 1) int

        out_feats = pointgroup_ops.voxelization(clusters_feats, out_map.cuda(), mode)  # (M, C), float, cuda

        spatial_shape = [fullscale] * 3
        voxelization_feats = spconv.SparseConvTensor(out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1)

        return voxelization_feats, inp_map, clusters_coords_mean0

    def parse_dynamic_params(self, params, out_channels):
        assert params.dim()==2
        assert len(self.weight_nums) == len(self.bias_nums)
        assert params.size(1) == sum(self.weight_nums) + sum(self.bias_nums)

        num_instances = params.size(0)
        num_layers = len(self.weight_nums)
        params_splits = list(torch.split_with_sizes(params, self.weight_nums+self.bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_instances*out_channels, -1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances*out_channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_instances, -1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_instances)

        return weight_splits, bias_splits



    def mask_heads_forward(self, mask_features, weights, biases, inst_batch_id, clusters_coords_mean, coords_, use_coords=True):
        num_insts = inst_batch_id.size(0)
        assert mask_features.dim() == 3
        n_layers = len(weights)
        c = mask_features.size(1)
        n_mask = mask_features.size(0)
        x = mask_features.permute(2,1,0).repeat(num_insts, 1, 1) ### num_inst * c * N_mask

        relative_coords = clusters_coords_mean.reshape(-1, 1, 3) - coords_.reshape(1, -1, 3) ### N_inst * N_mask * 3
        relative_coords = relative_coords.permute(0,2,1) ### num_inst * 3 * n_mask

        if use_coords:
            x = torch.cat([relative_coords, x], dim=1) ### num_inst * (3+c) * N_mask

        x = x.reshape(1, -1, n_mask) ### 1 * (num_inst*c') * Nmask
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv1d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)

        return x



    def get_instance_batch_id(self, batch_ids, inst_idx, inst_offsets):
        inst_num = inst_offsets.size(0) - 1
        inst_batch_id = torch.zeros(inst_num).int().cuda()
        for i in range(inst_num):
            start = inst_offsets[i].item()
            end = inst_offsets[i+1].item()
            pts_ids = inst_idx[start:end, 1]
            inst_batch_id[i] = batch_ids[pts_ids[0].long()]
            if batch_ids[pts_ids.long()].unique().size(0) > 1:
                assert RuntimeError
        return inst_batch_id

    def generate_proposal(self, mask_logits, batch_id, threshold, seg_pred, inst_pred_seg_label, min_pts_num=50):
        n_inst = mask_logits.size(0)
        proposal_len = []
        proposal_len.append(0)
        proposal_idx = []
        num = 0
        scores = []
        for n in range(n_inst):
            proposal_id_n = ((mask_logits[n] > threshold) + (seg_pred == inst_pred_seg_label[n].item()) > 1).nonzero().squeeze(dim=1)
            score = mask_logits[n][proposal_id_n].mean()
            seg_label = inst_pred_seg_label[n]
            if proposal_id_n.size(0) < min_pts_num:
                continue
            proposal_id_n = batch_id[proposal_id_n.long()].unsqueeze(dim=1)
            id_proposal_id_n = torch.cat([torch.ones_like(proposal_id_n)*num, proposal_id_n], dim=1)
            num += 1
            tmp = proposal_len[-1]
            proposal_len.append(proposal_id_n.size(0)+tmp)
            proposal_idx.append(id_proposal_id_n)
            scores.append(score)

        proposal_idx = torch.cat(proposal_idx, dim=0)
        proposal_len = torch.from_numpy(np.array(proposal_len)).cuda()
        scores = torch.stack(scores)
        # scores = torch.from_numpy(np.array(scores, dtype=np.float32)).cuda()
        return proposal_idx, proposal_len, scores


    def get_instance_seg_pred_label(self, semantic_label, proposals_idx, proposals_shift):
        instance_num = proposals_shift.size(0) - 1
        seg_labels = []
        for n in range(instance_num):
            start = proposals_shift[n].item()
            end = proposals_shift[n+1].item()
            ins_ids_n = proposals_idx[start:end, 1]
            seg_label_n = torch.mode(semantic_label[ins_ids_n.long()])[0].item()
            seg_labels.append(seg_label_n)

        return torch.from_numpy(np.array(seg_labels, dtype=np.int32)).cuda()





    def forward(self, input, input_map, coords, batch_idxs, batch_offsets, epoch, ins_sample_num=70, training=True):
        '''
        :param input_map: (N), int, cuda
        :param coords: (N, 3), float, cuda
        :param batch_idxs: (N), int, cuda
        :param batch_offsets: (B + 1), int, cuda
        '''
        ret = {}

        output = self.input_conv(input)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features[input_map.long()]
        output_feats = output_feats.contiguous()
        #### semantic segmentation
        semantic_scores = self.linear(output_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]    # (N), long

        ret['semantic_scores'] = semantic_scores

        #### offset
        pt_offsets_feats = self.offset(output_feats)
        pt_offsets = self.offset_linear(pt_offsets_feats)   # (N, 3), float32

        ret['pt_offsets'] = pt_offsets

        if(epoch > self.prepare_epochs):
            #### get prooposal clusters
            object_idxs = torch.nonzero(semantic_preds > 1).view(-1)

            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = utils.get_batch_offsets(batch_idxs_, input.batch_size)
            coords_ = coords[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            semantic_preds_ = semantic_preds[object_idxs]
            semantic_preds_cpu = semantic_preds_.int().cpu()

            idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
            proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
            proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()

            input_feats, inp_map, clusters_coords_mean = self.clusters_voxelization(proposals_idx_shift, proposals_offset_shift, output_feats, coords, self.score_fullscale, self.score_scale, self.mode)


            mask_features = self.mask_tower(torch.unsqueeze(output_feats, dim=2).permute(2,1,0)).permute(2,1,0)
            mask_features = mask_features[object_idxs]


            ### to generate weights
            params = self.score_unet(input_feats)
            params = self.score_outputlayer(params)
            params_feats = params.features[inp_map.long()] # (sumNPoint, C)
            params_feats = pointgroup_ops.roipool(params_feats, proposals_offset_shift.cuda())  # (nProposal, C)

            if len(params_feats) > ins_sample_num and ins_sample_num >0:
                params_feats = params_feats[:ins_sample_num]
                proposals_offset_shift = proposals_offset_shift[:ins_sample_num+1]
                clusters_coords_mean = clusters_coords_mean[:ins_sample_num]


            inst_batch_id = self.get_instance_batch_id(batch_idxs, proposals_idx_shift, proposals_offset_shift)
            inst_pred_seg_label = self.get_instance_seg_pred_label(semantic_preds, proposals_idx_shift, proposals_offset_shift)




            before_embedding_feature = self.before_embedding_tower(torch.unsqueeze(params_feats, dim=2))
            controller = self.controller(before_embedding_feature).squeeze(dim=2)

            weights, biases = self.parse_dynamic_params(controller, self.output_dim)


            ###
            ###


            # n_inst = len(params_feats)
            mask_logits = self.mask_heads_forward(mask_features, weights, biases, inst_batch_id, clusters_coords_mean, coords_, use_coords=self.use_coords)

            ret['mask_logits'] = mask_logits.squeeze(dim=0) ### N_inst * N_mask

            ret['object_idxs'] = object_idxs

            ret['proposals_offset_shift'] = proposals_offset_shift
            ret['proposals_idx_shift'] = proposals_idx_shift
            ret['inst_batch_id'] = inst_batch_id
            ret['batch_idxs'] = batch_idxs_
            ret['inst_pred_seg_label'] = inst_pred_seg_label






            if not training:
                ### generate proposal idx

                proposal_idx, proposal_len, scores = self.generate_proposal(mask_logits.squeeze(dim=0).sigmoid(), object_idxs,
                                                                            threshold=0.5, seg_pred=semantic_preds_, inst_pred_seg_label=inst_pred_seg_label,
                                                                            min_pts_num=50)

                ret['proposal_scores'] = (scores, proposal_idx, proposal_len)




        return ret


def model_fn_decorator(test=False):
    #### config
    from util.config import cfg

    #### criterion
    semantic_criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_label).cuda()
    score_criterion = nn.BCELoss(reduction='none').cuda()

    def test_model_fn(batch, model, epoch):
        coords = batch['locs'].cuda()              # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()  # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()          # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()          # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()  # (N, 3), float32, cuda
        feats = batch['feats'].cuda()              # (N, C), float32, cuda

        batch_offsets = batch['offsets'].cuda()    # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch, ins_sample_num=-1, training=False)
        semantic_scores = ret['semantic_scores']  # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']            # (N, 3), float32, cuda
        if (epoch > cfg.prepare_epochs):
            scores, proposals_idx, proposals_offset = ret['proposal_scores']

        ##### preds
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            if (epoch > cfg.prepare_epochs):
                preds['score'] = scores
                preds['proposals'] = (proposals_idx, proposals_offset)

        return preds


    def model_fn(batch, model, epoch):
        ##### prepare input and forward
        # batch {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
        # 'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
        # 'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
        # 'id': tbl, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}
        coords = batch['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = batch['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = batch['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = batch['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda

        coords_float = batch['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = batch['feats'].cuda()                          # (N, C), float32, cuda
        labels = batch['labels'].cuda()                        # (N), long, cuda
        instance_labels = batch['instance_labels'].cuda()      # (N), long, cuda, 0~total_nInst, -100

        instance_info = batch['instance_info'].cuda()          # (N, 9), float32, cuda, (meanxyz, minxyz, maxxyz)
        instance_pointnum = batch['instance_pointnum'].cuda()  # (total_nInst), int, cuda

        batch_offsets = batch['offsets'].cuda()                # (B + 1), int, cuda

        spatial_shape = batch['spatial_shape']

        if cfg.use_coords:
            feats = torch.cat((feats, coords_float), 1)
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map, cfg.mode)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, cfg.batch_size)

        ret = model(input_, p2v_map, coords_float, coords[:, 0].int(), batch_offsets, epoch)
        semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda



        loss_inp = {}
        loss_inp['semantic_scores'] = (semantic_scores, labels)
        loss_inp['pt_offsets'] = (pt_offsets, coords_float, instance_info, instance_labels)
        if(epoch > cfg.prepare_epochs):
            # scores, proposals_idx, proposals_offset = ret['proposal_scores']
            # loss_inp['proposal_scores'] = (scores, proposals_idx, proposals_offset, instance_pointnum)
            loss_inp['mask_logits'] = ret['mask_logits']
            loss_inp['object_idxs'] = ret['object_idxs']
            loss_inp['proposals_offset_shift'] = ret['proposals_offset_shift']
            loss_inp['proposals_idx_shift'] = ret['proposals_idx_shift']
            loss_inp['inst_batch_ids'] = ret['inst_batch_id']
            loss_inp['batch_idxs'] = ret['batch_idxs']
            loss_inp['batch_offsets'] = batch_offsets
            loss_inp['inst_pred_seg_label'] = ret['inst_pred_seg_label']



        loss, loss_out, infos = loss_fn(loss_inp, epoch)

        ##### accuracy / visual_dict / meter_dict
        with torch.no_grad():
            preds = {}
            preds['semantic'] = semantic_scores
            preds['pt_offsets'] = pt_offsets
            # if(epoch > cfg.prepare_epochs):
            #     preds['score'] = scores
            #     preds['proposals'] = (proposals_idx, proposals_offset)

            visual_dict = {}
            visual_dict['loss'] = loss
            for k, v in loss_out.items():
                visual_dict[k] = v[0]

            meter_dict = {}
            meter_dict['loss'] = (loss.item(), coords.shape[0])
            for k, v in loss_out.items():
                meter_dict[k] = (float(v[0]), v[1])

        return loss, preds, visual_dict, meter_dict


    def loss_fn(loss_inp, epoch):
        DEBUG = False

        loss_out = {}
        infos = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_inp['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        loss_out['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_labels = loss_inp['pt_offsets']
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        # instance_labels: (N), long

        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid = (instance_labels != cfg.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        loss_out['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_out['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        loss = cfg.loss_weight[0] * semantic_loss + cfg.loss_weight[1] * offset_norm_loss + cfg.loss_weight[2] * offset_dir_loss

        if (epoch > cfg.prepare_epochs):

            proposals_idx_shift = loss_inp['proposals_idx_shift']
            mask_logits = loss_inp['mask_logits']
            object_idxs = loss_inp['object_idxs']
            proposals_offset_shift = loss_inp['proposals_offset_shift']
            instance_masked = instance_labels[object_idxs]
            semantic_masked = semantic_labels[object_idxs]
            inst_batch_ids = loss_inp['inst_batch_ids']
            batch_ids = loss_inp['batch_idxs']
            batch_offsets = loss_inp['batch_offsets']
            inst_pred_seg_label = loss_inp['inst_pred_seg_label']

            inst_num = inst_batch_ids.size(0)
            inst_gt_mask = torch.zeros_like(mask_logits)
            weights = torch.zeros_like(mask_logits)


            if DEBUG:
                import numpy as np
                from util.draw_utils import write_ply_color, write_ply_rgb
                batch_size = batch_offsets.size(0) - 1
                sel_batch_id = 0
                batch_start = batch_offsets[sel_batch_id].item()
                batch_end = batch_offsets[sel_batch_id+1].item()
                coords_np = coords.cpu().numpy()[batch_start:batch_end]
                ins_label_np = instance_labels.cpu().numpy()[batch_start:batch_end]
                m = np.max(ins_label_np)
                ins_label_np[ins_label_np < 0] = m + 1

                seg_label_np = semantic_labels.cpu().numpy()[batch_start:batch_end]
                m = np.max(seg_label_np)
                seg_label_np[seg_label_np < 0] = m + 1

                write_ply_color(coords_np, ins_label_np, 'ins.obj')
                write_ply_color(coords_np, seg_label_np, 'seg.obj')


            assert inst_pred_seg_label.size(0) == inst_num
            for n in range(inst_num):
                start = proposals_offset_shift[n].item()
                end = proposals_offset_shift[n+1].item()
                ins_ids_n = proposals_idx_shift[start:end, 1]
                ins_label_n = torch.mode(instance_labels[ins_ids_n.long()])[0].item()
                seg_label_n_pred = inst_pred_seg_label[n].item()
                inst_n_batch_id = int(inst_batch_ids[n].item())
                weights[n, batch_ids==inst_n_batch_id] = 1


                if ins_label_n < 0 or seg_label_n_pred <= 1:
                    continue
                weights[n,semantic_masked!=seg_label_n_pred] = 0



                batch_start = batch_offsets[inst_n_batch_id].item()
                batch_end = batch_offsets[inst_n_batch_id+1].item()
                cover_percent = (instance_masked[batch_ids==inst_n_batch_id] == ins_label_n).float().sum() /\
                                (instance_labels[batch_start:batch_end] == ins_label_n).float().sum()

                if cover_percent > 0.3:
                    inst_gt_mask[n] = (instance_masked ==ins_label_n)


                if DEBUG:
                    sel_batch_id = 0
                    if inst_n_batch_id != sel_batch_id:
                        print('batch id not match')
                        continue
                    coords_ins_np = coords[ins_ids_n.long()].cpu().numpy()
                    write_ply_rgb(coords_ins_np, np.ones_like(coords_ins_np), 'proposal_pts_{}.obj'.format(n))


                    batch_start = batch_offsets[sel_batch_id].item()
                    batch_end = batch_offsets[sel_batch_id+1].item()
                    ins_label_np2 = instance_labels.cpu().numpy()[batch_start:batch_end]
                    coord_proposal_n_gt = coords.cpu().numpy()[batch_start:batch_end][ins_label_np2==ins_label_n]
                    write_ply_rgb(coord_proposal_n_gt, np.zeros_like(coord_proposal_n_gt), 'proposal_pts_{}_gt.obj'.format(n))

                    inst_gt_mask_n = inst_gt_mask[n].cpu().numpy()
                    coords_np = coords[object_idxs.long()][batch_ids==sel_batch_id].cpu().numpy()
                    write_ply_color(coords_np, inst_gt_mask_n[batch_ids.cpu().numpy()==sel_batch_id], 'proposal_pts_{}_gt_mask.obj'.format(n))

                    coords_np2 = coords[object_idxs][batch_ids==sel_batch_id].cpu().numpy()
                    weights_np2 = weights[n][batch_ids==sel_batch_id].cpu().numpy()
                    write_ply_color(coords_np2, weights_np2, 'weight_{}.obj'.format(n))
                    print(weights[n][batch_ids!=sel_batch_id].sum())

                    pred_maks = (torch.sigmoid(mask_logits[n].view(-1)) > 0.5).cpu().numpy()
                    write_ply_color(coords_np, pred_maks[batch_ids.cpu().numpy()==sel_batch_id], 'proposal_pts_{}_pred_mask.obj'.format(n))

            score_loss = score_criterion(torch.sigmoid(mask_logits.view(-1)), inst_gt_mask.view(-1))
            score_loss = (score_loss* weights.view(-1)).sum() / (weights.sum() + 1e-6)
            score_loss = score_loss.mean()
            loss_out['score_loss'] = (score_loss, proposals_offset_shift.size(0)-1)
            loss += (cfg.loss_weight[3] * score_loss)



        return loss, loss_out, infos




    if test:
        fn = test_model_fn
    else:
        fn = model_fn
    return fn

