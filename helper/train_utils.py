import torch
from ogb.nodeproppred import Evaluator
from torch.optim.lr_scheduler import  _LRScheduler
from torch_geometric.utils import index_to_mask, subgraph
from model.nn import get_model
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from helper.utils import *
import os.path as osp
def pre_train(data_all, seed, args, device='cuda'):
    # reset_args(args)
    # 根据不同的model建立backbone
    model = get_model(args).to(device)
    filename = f"D:/code/LLM/LLMTTT/pretrain_goodgcn_val/{args.dataset}_{args.model_name}_s0.pt"
    if args.debug and osp.exists(filename):
        model.load_state_dict((torch.load(filename, map_location=device)))
    else:
        train_iters = 500 if args.dataset in ['arxiv', 'products'] else 1000
        #inductive：model.fit_inductive(data_all, train_iters=train_iters, patience=500, verbose=True)
        model.fit_transductive(data_all, train_iters=train_iters, patience=500, verbose=True)
        if args.debug:
            torch.save(model.state_dict(), filename)
    return model

def to_inductive(data, msk_index = 0):
    data = data.clone()
    # 选择索引为msk_index的掩码数组
    mask = data.train_masks[msk_index]
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = mask[mask]
    data.test_masks = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data

def graph_consistency(neighbors, gt):
    single_label_consistency = {}
    for key, value in neighbors.items():
        consistency = 0
        total_nei = len(value)
        center_y = gt[key]
        ## key: nodes
        ## value: neighbors
        for nei in value:
            nei_y = gt[nei]
            if nei_y == center_y:
                consistency += 1
        if total_nei != 0:
            single_label_consistency[key] = consistency / total_nei
        else:
            single_label_consistency[key] = 0
    sorted_keys = sorted(single_label_consistency, key=single_label_consistency.get)
    return sorted_keys

#训练模型的函数
# 损失函数的计算方式是根据传入的loss_fn计算训练损失train_loss，
# 如果提供了可信度列表reliability_list，则根据可信度进行加权计算。
# 然后对train_loss进行反向传播并更新模型参数。
def train(model, data, optimizer, loss_fn, train_mask, val_mask, no_val, reliability_list = None):
# reliability_list 可信度列表
    # 梯度置零
    optimizer.zero_grad()
    # 使用模型对数据进行预测
    preds = model(data)
    if len(data.y.shape) != 1:
        y = data.y.squeeze(1)
    else:
        y = data.y
    confidence = reliability_list
    # 如果 reduction 属性不是 'none' 或者没有提供 reliability_list
    if loss_fn.reduction != 'none' or confidence == None:
        # 直接计算训练损失train_loss
        train_loss = loss_fn(preds[train_mask], y[train_mask])
    # reduction属性是None且提供reliability_list
    else:
        # Extract the values using the mask 归一化处理 提取出对应位置的可信度值
        values_to_normalize = confidence[train_mask]
        # Compute min and max of these values
        min_val = torch.min(values_to_normalize)
        max_val = torch.max(values_to_normalize)
        # 最小值等于最大值 → 不用处理
        if min_val == max_val:
            # normalized_values = confidence
            pass
        # Apply Min-Max scaling
        else:
            # 将可信度值进行 Min-Max 缩放，使其范围在 [0, 1] 内
            normalized_values = (values_to_normalize - min_val) / (max_val - min_val)
            # Replace original tensor values with the normalized values for nodes defined by the train mask
            # 将 train_mask 对应位置的可信度值替换为归一化后的值
            confidence[train_mask] = normalized_values.clone()
        # 加权后的训练损失
        train_loss = (loss_fn(preds[train_mask], y[train_mask]) * confidence[train_mask]).mean()
    train_loss.backward()
    optimizer.step()
    # 计算训练集上准确率
    train_acc, _ = test(model, data, False, train_mask)
    if not no_val:
        val_loss = loss_fn(preds[val_mask], y[val_mask])
        val_acc, _ = test(model, data, False, val_mask)
    else:
        val_loss = 0
        val_acc = 0
    return train_loss, val_loss, val_acc, train_acc


@torch.no_grad()
def test(model, data, return_embeds, mask, gt_y=None):
    model.eval()
    # model.model.initialized = False
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if gt_y != None:
        if len(gt_y.shape) == 1:
            y = gt_y.unsqueeze(dim=1)  # for non ogb datas
        else:
            y = gt_y
    else:
        if len(data.y.shape) == 1:
            y = data.y.unsqueeze(dim=1)
        else:
            y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    acc = evaluator.eval({
        'y_true': y[mask],
        'y_pred': y_pred[mask],
    })['acc']
    # acc = model.eval_func(y, y_pred, mask)
    if not return_embeds:
        # return acc.cpu().item(), None
        return acc, None
    else:
        return acc.cpu().item(), out

def evaluate_transductive(data_all, dataset, model, device="cuda"):
    model.eval()
    accs = []
    y_te, out_te = [], []
    out = model(data_all)
    y_pred = out.argmax(dim=-1, keepdim=True)
    if len(data_all.y.shape) == 1:
        y = data_all.y.unsqueeze(dim=1)
    else:
        y = data_all.y

def evaluate(data_all, dataset, model, device="cuda"):
    # 通知所有层，处于评估模式，声明模型状态
    model.eval()
    accs = []
    y_te, out_te = [], []
    y_te_all, out_te_all = [], []
    for ii, test_data in enumerate(data_all[2]):
        x, edge_index = test_data.graph['node_feat'], test_data.graph['edge_index']
        x, edge_index = x.to(device), edge_index.to(device)
        output = model.predict(x, edge_index)

        labels = test_data.label.to(device)  # .squeeze()
        eval_func = model.eval_func
        if dataset in ['ogb-arxiv']:
            acc_test = eval_func(labels[test_data.test_mask], output[test_data.test_mask])
            accs.append(acc_test)
            y_te_all.append(labels[test_data.test_mask])
            out_te_all.append(output[test_data.test_mask])
        elif dataset in ['cora', 'amazon-photo', 'twitch-e', 'fb100']:
            acc_test = eval_func(labels, output)
            accs.append(acc_test)
            y_te_all.append(labels)
            out_te_all.append(output)
        elif dataset in ['elliptic']:
            acc_test = eval_func(labels[test_data.mask], output[test_data.mask])
            y_te.append(labels[test_data.mask])
            out_te.append(output[test_data.mask])
            y_te_all.append(labels[test_data.mask])
            out_te_all.append(output[test_data.mask])
            if ii % 4 == 0 or ii == len(data_all[2]) - 1:
                acc_te = eval_func(torch.cat(y_te, dim=0), torch.cat(out_te, dim=0))
                accs += [float(f'{acc_te:.2f}')]
                y_te, out_te = [], []
        else:
            raise NotImplementedError
    print('Test accs:', accs)
    acc_te = eval_func(torch.cat(y_te_all, dim=0), torch.cat(out_te_all, dim=0))
    print(f'flatten test: {acc_te}')
    return acc_te
class WarmupExpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, gamma=0.1, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.gamma = gamma
        super(WarmupExpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * self.gamma
                    for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]

def get_optimizer(args, model):
    if args.model_name == 'LP':
        return None, None
    if args.optim == 'adam':
        optimizer_1 = torch.optim.Adam(model.parameters(), lr = args.lr_ttt_1, weight_decay=args.weight_decay)
        optimizer_2 = torch.optim.Adam(model.parameters(), lr = args.lr_ttt_2, weight_decay=args.weight_decay)
        scheduler = None
    elif args.optim == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr = args.lr_ttt, weight_decay=args.weight_decay)
        scheduler = WarmupExpLR(optimizer, args.warmup, total_epochs=args.epochs, gamma=args.lr_gamma)
    return optimizer_1, optimizer_2, scheduler