import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
import json
import torch
import numpy as np
from lfh import HHGNN
import os
import pickle
import collections


def process_train_data(node_info, feature, num_cat):
    feature = torch.from_numpy(feature).to(torch.float32)
    for key in ["train_label", "val_label", "test_label"]:
        node_info[key] = torch.from_numpy(np.eye(num_cat)[node_info[key]])
    return feature, node_info


def process_label(label):
    train_idx = [ele[0] for ele in label[0]]
    val_idx = [ele[0] for ele in label[1]]
    test_idx = [ele[0] for ele in label[2] if ele[0] not in [76,122,64]]
    train_label = [ele[1] for ele in label[0]]
    val_label = [ele[1] for ele in label[1]]
    test_label = [ele[1] for ele in label[2] if ele[0] not in [76,122,64]]
    return dict(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, train_label=train_label, val_label=val_label,
                test_label=test_label)


def node_idx_type_update(node_idx_r_dict, update):
    if update == dict():
        return node_idx_r_dict
    key = list(update.keys())[0]
    now_key_list = list(node_idx_r_dict.keys())
    if key in now_key_list:
        if key > 3024:
            assert 0, "0-3024 relation error"
    else:
        node_idx_r_dict.update(update)
    return node_idx_r_dict


def process_edges(edges, node_idx_r_dict, sub=True):
    num_r = len(edges)
    num_node = edges[0].shape[0]
    adj_dict = collections.defaultdict(dict)
    if node_idx_r_dict == None:
        node_idx_r_dict = dict()
    data_stat = dict()

    for i in range(num_node):
        adj_dict[i] = dict()

    if sub:
        for i in tqdm(range(num_r)):
            edge = edges[i].A
            for j in range(num_node):
                s_node = j
                t_node_list = np.nonzero(edge[s_node])[0].tolist()
                adj_dict[s_node].update({i: t_node_list})
    else:
        for i in tqdm(range(num_r)):
            edge = edges[i].A
            for j in range(num_node):
                s_node = j
                t_node_list = np.nonzero(edge[s_node])[0].tolist()
                adj_dict[s_node].update({i: t_node_list})
                node_idx_r_dict = node_idx_type_update(node_idx_r_dict, get_node_idx_type(t_node_list, j, i))
            del edge

    # acm special process
    node_idx_r_dict.update({k: 2 for k, v in node_idx_r_dict.items() if v == 3})
    data_stat['node_idx_r_map'] = node_idx_r_dict
    data_stat.update(get_info(node_idx_r_dict))

    return adj_dict, data_stat


def get_info(node_idx_r_dict):
    num_r_node_stat_dict = collections.defaultdict(list)
    m_node_r_slave_node = dict()
    for k, v in node_idx_r_dict.items():
        num_r_node_stat_dict[v].append(k)
    num_node = len(list(node_idx_r_dict.keys()))
    num_r = len(list(num_r_node_stat_dict.keys()))
    all_node = len(list(node_idx_r_dict.keys()))

    r_list = list(num_r_node_stat_dict.keys())
    for i in range(all_node):
        master_node_type = node_idx_r_dict[i]
        m_node_r_slave_node[i] = dict()
        for j in r_list:
            if master_node_type != j:
                m_node_r_slave_node[i][j] = num_r_node_stat_dict[j]
            else:
                slave_node_idx_list = list(num_r_node_stat_dict[j])
                slave_node_idx_list.remove(i)
                m_node_r_slave_node[i][j] = slave_node_idx_list

    num_r_neigh = dict()
    for i in range(num_r):
        num_r_neigh[i] = dict()
        for j in range(num_r):
            if i == j:
                num_r_neigh[i][j] = len(num_r_node_stat_dict[j]) - 1
            else:
                num_r_neigh[i][j] = len(num_r_node_stat_dict[j])
    return {"num_r_node_stat": num_r_node_stat_dict, "num_node": num_node, "num_r": num_r, "m_r_s": m_node_r_slave_node,
            "num_r_neigh": num_r_neigh}


def get_node_idx_type(t_node_list, node_idx, r):
    if r == 2:
        r = 0
    if len(t_node_list) > 0:
        return {node_idx: r}
    else:
        return dict()


def process_clean_version(adj_dict, data_stat, num_cat):
    rlt = dict()
    rlt['num_node'] = data_stat['num_node']
    rlt['num_type'] = data_stat['num_r']
    rlt['num_cat'] = num_cat
    rlt['num_head'] = 4
    rlt['num_fea'] = 256
    rlt['mask'], rlt['edge_mask'], rlt['node_multi_mask'], rlt['edge_multi_mask'] = return_mask(adj_dict, data_stat)

    return rlt


def return_mask(adj_dict, data_stat):
    # candidate slave node -->1 ,others --> 0
    num_node = data_stat['num_node']
    num_type = data_stat['num_r']
    n_r_map = data_stat['node_idx_r_map']
    type_node_idx_span = data_stat['num_r_node_stat']
    candidate_mask = torch.zeros(num_node, num_type, num_node)
    edge_mask = torch.zeros(num_node, num_type, 256, dtype=torch.int64)

    for node_idx in list(adj_dict.keys()):
        master_node_type = n_r_map[node_idx]
        for type_idx in range(num_type):
            candidate_slave_node_id_list = list(type_node_idx_span[type_idx])
            if node_idx in candidate_slave_node_id_list:
                candidate_slave_node_id_list.remove(node_idx)
            candidate_mask[node_idx][type_idx][candidate_slave_node_id_list] = 1
            edge_mask[node_idx][type_idx][:] = master_node_type * num_type + type_idx
    node_multi_mask = {i: n_r_map[i] for i in range(num_node)}
    edge_multi_mask = dict()

    return candidate_mask.unsqueeze(dim=2), edge_mask, node_multi_mask, edge_multi_mask

        
def prepare(ds, sub=True, ratio=0.3, sample=None):
    # existing data
    if os.path.exists("self_data/{}4GTN_{}/split_info.pkl".format(ds, sample)):
        with open('self_data/{}4GTN_{}/edges.pkl'.format(ds, sample), 'rb') as f:
            obj = f.read()
            edges = pickle.loads(obj, encoding='latin1')
        with open('self_data/{}4GTN_{}/labels.pkl'.format(ds, sample), 'rb') as f:
            obj = f.read()
            labels = pickle.loads(obj, encoding='latin1')
        with open('self_data/{}4GTN_{}/node_features.pkl'.format(ds, sample),
                  'rb') as f:
            obj = f.read()
            node_features = pickle.loads(obj, encoding='latin1')
        with open('self_data/{}4GTN_{}/split_info.pkl'.format(ds, sample),
                  'rb') as f:
            obj = f.read()
            node_idx_r_dict = pickle.loads(obj, encoding='latin1')
        print("have data already!")

    target_node_info = process_label(labels)

    if ds == "dblp":
        num_cat = 4
    else:
        num_cat = 3

    adj_dict, data_stat = process_edges(edges, node_idx_r_dict, sub)
    data_stat = process_clean_version(adj_dict, data_stat, num_cat)
    with open("label.json", "w") as p:
        json.dump(node_idx_r_dict, p, indent=4)
    return target_node_info, node_features, data_stat


def run(node_info, feature, data_stat):
    lr = 0.0020
    epoch = 100

    torch.manual_seed(13956373134037443673)

    feature, node_info = process_train_data(node_info, feature, data_stat['num_cat'])

    models = HHGNN(data_stat)
    print("hypergraph trainning!")

    optimizer = optim.Adam(models.parameters(), lr=lr, weight_decay=0.01)
    train_idx = node_info['train_idx']
    train_label = node_info['train_label']
    val_idx = node_info['val_idx']
    val_label = node_info['val_label']
    test_idx = node_info['test_idx']
    test_label = node_info['test_label']
    metric = F.binary_cross_entropy_with_logits
    best_acc = -1
    best_f1_mi = -1
    best_f1_ma = -1
    # feature = torch.load("self_data/{}4GTN_{}/node_features.pickle".format(ds, parameter['sample'], ds))
    with open('self_data/acm4GTN_10/node_features.pkl',
                  'rb') as f:
            obj = f.read()
            feature = pickle.loads(obj, encoding='latin1')
    feature = torch.from_numpy(feature).float()

    for i in tqdm(range(epoch)):
        models = models.train()
        optimizer.zero_grad()
        pred, recons_loss, node_rep = models(feature, train_idx + val_idx, data_stat)
        train_loss = metric(pred, torch.vstack((train_label, val_label)), reduction='sum')

        print("label_loss:{},recon_loss:{}".format(train_loss, recons_loss))

        train_loss = train_loss + recons_loss
        train_loss.backward()
        optimizer.step()
        models = models.eval()
        pred = models.pred(node_rep[test_idx])
        val_loss = metric(pred, test_label, reduction='sum')
        _, preds = torch.max(pred, 1)
        _, gold = torch.max(test_label, 1)
        acc = torch.mean((preds == gold).float())
        not_same = [test_idx[i] for i in range(preds.shape[0]) if preds[i] != gold[i]]
        print(not_same)
        f1_mi = f1_score(preds, gold, average='micro')
        f1_ma = f1_score(preds, gold, average='macro')
        print("epoch:{},train_loss:{},val_loss:{},acc:{},f1_micro:{},f1_macro:{}".format(i, train_loss, val_loss, acc,
                                                                                         f1_mi, f1_ma))
        if f1_ma > best_f1_ma:
            best_f1_ma = f1_ma
        if f1_mi > best_f1_mi:
            best_f1_mi = f1_mi
            # np.save("fea.npy",node_rep.clone().detach().numpy())
        if acc > best_acc:
            best_acc = acc
    print("best:{}".format(best_f1_mi))

def main():
    node_info, feature, data_stat = prepare(ds, sub=parameter['sub'],
                                            ratio=float(parameter['sample']) / 100,
                                            sample=parameter['sample'])
    run(node_info, feature, data_stat)


parameter = {
    "sub": True,
    "sample": "10"
}
ds = 'acm'
main()

if __name__ == "__main__":
    main()
