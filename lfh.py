import math
import torch
import torch.nn as nn



class HHEdgeCons(nn.Module):
    # lambda from elastic net
    def __init__(self, data_stat, recons_error_lambda=0.1, l2_lambda=0.2, recons_lambda=0.01):
        super(HHEdgeCons, self).__init__()
        self.l2_lambda = l2_lambda
        self.recons_lambda = recons_lambda
        self.num_node = data_stat['num_node']
        self.num_type = data_stat['num_type']
        self.num_fea = data_stat['num_fea']
        self.recons_error_lambda = recons_error_lambda
        self.linear = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.rand(self.num_node, self.num_type, 1, self.num_node)))

        self.recon_original_proj = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.rand(self.num_type * self.num_type, self.num_fea, self.num_fea)))


    def node_project(self, fea, proj, node_multi_mask):
        return torch.vstack(
            [torch.matmul(proj[node_multi_mask[i] * self.num_type + j], fea[i][j]) for i in range(self.num_node)
             for j in range(self.num_type)]).reshape(self.num_node, self.num_type, self.num_fea)

    def forward(self, feature, mask, node_multi_mask):
        feature_frozen = feature.detach()
        inp_4_recon = feature_frozen.expand(self.num_type, self.num_node, self.num_fea)
        # different candidate slave nodes
        linear_mask = self.linear * mask
        # selected slave nodes with reconstruction value larger than 0
        l0_mask = linear_mask > 0
        linear_selected = linear_mask.masked_fill(~l0_mask, value=torch.tensor(0))
        reconstruct_fea = torch.matmul(linear_selected, inp_4_recon).squeeze()

        inp_original = torch.transpose(inp_4_recon, 1, 0)

        inp_projected_selected = self.node_project(inp_original, self.recon_original_proj, node_multi_mask)
        # reconstruction error
        linear_comb_l1 = torch.sum(torch.norm(linear_selected.squeeze(), dim=-1, p=1))
        linear_comb_l2 = torch.sum(torch.norm(linear_selected.squeeze(), dim=-1, p=2))
        recon_error = torch.sum(torch.norm(inp_projected_selected - reconstruct_fea, dim=-1, p=2))
        recon_loss = self.recons_lambda * recon_error + self.l2_lambda * linear_comb_l2 + linear_comb_l1
        return linear_selected, self.recons_error_lambda * recon_loss


class HHEdgeMP(nn.Module):
    def __init__(self, data_stat):
        super(HHEdgeMP, self).__init__()
        self.num_node = data_stat['num_node']
        self.num_type = data_stat['num_type']
        self.num_fea = data_stat['num_fea']

    def forward(self, feature, linear_selected):
        edge_fea = linear_selected @ feature
        edge_fea = edge_fea.squeeze() + torch.transpose(feature.expand(self.num_type, self.num_node, self.num_fea), 1,
                                                        0)
        # edge norm
        edge_norm = torch.sum(linear_selected.squeeze(), dim=-1) + torch.ones(self.num_node, self.num_type)
        edge_fea = torch.div(edge_fea, edge_norm.unsqueeze(dim=2))
        return edge_fea


class MultiheadWeight(nn.Module):
    def __init__(self, data_stat):
        super(MultiheadWeight, self).__init__()
        self.num_node = data_stat['num_node']
        self.num_type = data_stat['num_type']
        self.num_fea = data_stat['num_fea']
        self.num_head = data_stat['num_head']

        self.multi_head_node_proj = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.rand(self.num_type, self.num_fea, self.num_fea)))
        self.multi_head_edge_proj = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.rand(self.num_type * self.num_type, self.num_fea, self.num_fea)))

        self.act_1 = nn.Softmax(dim=1)

    def node_project(self, fea, proj, node_multi_mask):
        return torch.vstack([torch.matmul(proj[node_multi_mask[i]], fea[i]) for i in range(self.num_node)])

    def edge_project(self, fea, proj, node_multi_mask):
        return torch.vstack(
            [torch.matmul(proj[node_multi_mask[i] * self.num_type + j], fea[i][j]) for i in range(self.num_node)
             for j in range(self.num_type)])

    def forward(self, feature, edge_fea, node_multi_mask):
        node_m_projected = self.node_project(feature, self.multi_head_node_proj, node_multi_mask)

        edge_m_projected = self.edge_project(edge_fea, self.multi_head_edge_proj, node_multi_mask)

        node_multi = node_m_projected.reshape(self.num_node, 1, self.num_head, int(self.num_fea / self.num_head))
        edge_multi = edge_m_projected.reshape(self.num_node, self.num_type, self.num_head,
                                              int(self.num_fea / self.num_head))

        edge_multi_in = edge_multi.permute(0, 2, 3, 1)
        node_multi_in = node_multi.permute(0, 2, 1, 3) / math.sqrt(
            int(self.num_fea / self.num_head))
        all_r_weight = self.act_1(torch.matmul(node_multi_in, edge_multi_in)).permute(0, 3, 1, 2)

        return all_r_weight


class HHNodeMP(nn.Module):
    def __init__(self, data_stat, drop_rate=0.3):
        super(HHNodeMP, self).__init__()
        self.num_node = data_stat['num_node']
        self.num_type = data_stat['num_type']
        self.num_fea = data_stat['num_fea']
        self.num_head = data_stat['num_head']
        self.drop = nn.Dropout(drop_rate)
        self.act = nn.ReLU(inplace=True)

    def forward(self, edge_fea, all_r_weight):
        edge_fea_mp = edge_fea.reshape(self.num_node, self.num_type, self.num_head, int(self.num_fea / self.num_head))
        edge_fea_weighted = torch.mul(all_r_weight, edge_fea_mp)

        node_rep = self.drop(self.act(edge_fea_weighted.reshape(self.num_node, self.num_type, self.num_fea)))
        node_rep = torch.sum(node_rep, dim=1)
        return node_rep


class Predictor(nn.Module):
    def __init__(self, data_stat):
        super(Predictor, self).__init__()
        self.num_fea = data_stat['num_fea']
        self.num_cat = data_stat['num_cat']
        self.predict = nn.Linear(self.num_fea, self.num_cat)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        self.sigma = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, node_emb):
        n_cat = self.predict(node_emb)
        n_cat = self.sigma(n_cat)
        n_cat = self.softmax(n_cat).squeeze()
        return n_cat


class HHGNN(nn.Module):
    def __init__(self, data_stat):
        super(HHGNN, self).__init__()
        self.hhedgecons = HHEdgeCons(data_stat)
        self.hhedgemp = HHEdgeMP(data_stat)
        self.mheadweight = MultiheadWeight(data_stat)
        self.hhnodemp = HHNodeMP(data_stat)
        self.num_fea = data_stat['num_fea']
        self.theta = nn.Linear(self.num_fea, self.num_fea, bias=True)
        self.pred = Predictor(data_stat)
        torch.nn.init.xavier_uniform_(self.theta.weight)

    def forward(self, feature, node_idx, data_stat):
        feature = self.theta(feature)
        linear_selected, recon_loss = self.hhedgecons(feature, data_stat['mask'], data_stat['node_multi_mask'])
        edge_fea = self.hhedgemp(feature, linear_selected)

        all_r_weight = self.mheadweight(feature, edge_fea, data_stat['node_multi_mask'])

        node_rep = self.hhnodemp(edge_fea, all_r_weight)

        predict = self.pred(node_rep[node_idx])
        return predict, recon_loss, node_rep
