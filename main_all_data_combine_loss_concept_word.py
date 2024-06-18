import torch
from helper.utils import *
import argparse
from tqdm import tqdm
from helper.args import get_command_line_args, replace_args_with_dict_values
from helper.train_utils import get_optimizer, pre_train, test
import torch.nn.functional as F
from copy import deepcopy
import time
from helper.data import get_dataset, load_raw_files, load_ood, load_llm_pred_me, load_llm_pred_TAPE, generate_test_mask,active_selection
from model.nn import get_model
import pandas as pd
import torch.nn as nn
import optuna
import os.path as osp
args = get_command_line_args()
seeds = [i for i in range(args.main_seed_num)]
param_range = list(seeds)

def get_new_model(model, changed_model, fixed_model, changed, nlayers):
    if changed == nlayers:
        return model
    else:
        for j in range(changed):
            model.convs[j] = changed_model[j]
        for i in range(changed, nlayers):
            model.convs[i] = fixed_model[0][i-changed]
            if i == nlayers - 1:
                return model
            else:
                model.norms[i] = fixed_model[1][i-changed]
    return model
def c_train(model, data, optimizer, loss_fn, train_mask, val_mask, no_val, llm_pred, conf):
    model.train()
    conf = conf.to('cuda')
# reliability_list 可信度列表
    # 梯度置零
    optimizer.zero_grad()
    # 使用模型对数据进行预测
    preds = model(data)
    if loss_fn != None:
        train_loss = loss_fn(preds[train_mask], data.y[train_mask].long())
        train_loss.backward()
        optimizer.step()
        # 计算训练集上准确率
        train_acc, _ = test(model, data, False, train_mask)
        val_loss = loss_fn(preds[val_mask], llm_pred[val_mask])
        val_acc, _ = test(model, data, False, val_mask)
        return train_loss, val_loss, val_acc, train_acc
    else:
        probs = torch.softmax(preds, dim=-1)
        mask = soft_match_weighting(probs, data.y.max()+1)
        pseudo_label = torch.argmax(preds, dim=-1)
        unsup_loss = consistency_loss(preds[train_mask], pseudo_label[train_mask], 'ce', mask[train_mask])
        train_loss = unsup_loss
        train_loss.backward()
        optimizer.step()
        # 计算训练集上准确率
        train_acc, _ = test(model, data, False, train_mask)
        unsup_loss = consistency_loss(preds[val_mask], llm_pred[val_mask], 'ce', mask[val_mask])
        val_loss = unsup_loss
        val_acc, _ = test(model, data, False, val_mask)
        return train_loss.cpu().item(), val_loss.cpu().item(), val_acc, train_acc
def train(model, data, optimizer, loss_fn, train_mask, val_mask, no_val, llm_pred, conf):
    model.train()
    conf = conf.to('cuda')
# reliability_list 可信度列表
    # 梯度置零
    optimizer.zero_grad()
    # 使用模型对数据进行预测
    preds = model(data)
    if loss_fn != None:
        confidence = conf
        # ce
        if loss_fn.reduction != 'none' or confidence == None:
            # 直接计算训练损失train_loss
            # train_loss = loss_fn(preds[train_mask], llm_pred[train_mask].long())
            train_loss = loss_fn(preds[train_mask], data.y[train_mask].long())
        # rim
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
            train_loss = (loss_fn(preds[train_mask], llm_pred[train_mask]) * confidence[train_mask]).mean()
        train_loss.backward()
        optimizer.step()
        # 计算训练集上准确率
        train_acc, _ = test(model, data, False, train_mask)
        val_loss = loss_fn(preds[val_mask], llm_pred[val_mask])
        val_acc, _ = test(model, data, False, val_mask)
        return train_loss, val_loss, val_acc, train_acc
    else:
        mask = soft_match_weighting(conf[train_mask], data.y.max()+1)
        unsup_loss = consistency_loss(preds[train_mask], llm_pred[train_mask], 'ce', mask)
        train_loss = unsup_loss
        train_loss.backward()
        optimizer.step()
        # 计算训练集上准确率
        train_acc, _ = test(model, data, False, train_mask)
        # sup_loss = ce_loss(preds[train_mask], data.y[train_mask])
        ce = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        sup_loss = ce(preds[val_mask], data.y[val_mask])
        # uniform distribution alignment
        mask = soft_match_weighting(conf[val_mask], data.y.max()+1)
        lambda_u = 1
        unsup_loss = consistency_loss(preds[val_mask], llm_pred[val_mask], 'ce', mask)
        val_loss = unsup_loss
        val_acc, _ = test(model, data, False, val_mask)
        return train_loss.cpu().item(), val_loss.cpu().item(), val_acc, train_acc
def get_changed_gnn_list(model, changed, nlayers):
    # 判断共享层合理性
    if changed > nlayers:
        print("Warning: You Want to Share Too Many Layers, Cut Down to Depth.")
        changed = nlayers
    elif changed < 0:
        raise NotImplementedError(f"At Least Share One Layer Please.")
        # 得到共享层需要改变的参数 name list 等号相当于浅拷贝，module list就相当于需要更改的参数的 name list
    module_list = nn.ModuleList()
    for j in range(changed):
        con = deepcopy(model.convs[j])
        module_list.append(con)
    # print(module_list)
    head_list_bn = nn.ModuleList()  # nothing when changed_num == layers
    head_list_con = nn.ModuleList()  # nothing when changed_num == layers
    for i in range(changed, nlayers):
        head_convs = deepcopy(model.convs[i])
        if i == nlayers - 1:
            head_bns = None
        else:
            head_bns = deepcopy(model.norms[i])
        head_list_bn.append(head_bns)
        head_list_con.append(head_convs)
    head_list = [head_list_con, head_list_bn]
    return module_list, head_list
def test_time_training(model, seed, args, data, llm_pred, conf):
    #model.train()
    # 选择GPU还是CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化变量
    early_stop_accum = 0
    best_val = 0
    best_model = None
    epoch = args.epochs
    need_train = True
    seed_everything(seed)
    # 选择模型
    model = model.to(device)
    llm_pred = llm_pred.to(device)
    changed_model, fixed_model = get_changed_gnn_list(model, args.changed, args.num_layers)
    # 根据参数选择损失函数 → 交叉熵损失函数
    if args.loss_type == 'ce':
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    elif args.loss_type == 'rim':
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='none')
    elif args.loss_type == 'trade_off':
        loss_fn = None
    # 归一化处理
    if args.normalize:
        data.x = F.normalize(data.x, dim = -1)
    data = data.to(device)
    # 训练集和测试集的掩码选择数据
    debug_acc = []
    this_train_acc = []
    train_mask = data.train_masks[seed]
    val_mask = data.val_masks[seed]
    test_mask = data.test_masks[seed]
    print("train_num: %s; val_num: %s; test_num: %s" % (train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item()))
    acc = (llm_pred[train_mask] == data.y[train_mask]).float().mean()
    print("LLM acc on train_data:", acc)
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=False)
    acc = (y_pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    print("GCN acc on all test data:", acc)
    acc = (y_pred[train_mask] == data.y[train_mask]).float().mean()
    print("GCN acc on test train data:", acc)
    #return model
    data.backup_y = data.y.clone()
    precisions_1 = []
    precisions_2 = []
    precisions_3 = []
    precisions_4 = []
    losses = []
    mask = torch.Tensor([True for i in range(data.x.shape[0])]).bool()
    test_acc, res = test(model, data, 0, train_mask)
    precisions_1.append(test_acc)
    test_acc, res = test(model, data, 0, test_mask)
    precisions_2.append(test_acc)
    test_acc, res = test(model, data, 0, data.test_mask)
    precisions_3.append(test_acc)
    test_acc, res = test(model, data, 0, val_mask)
    precisions_4.append(test_acc)
    for i in tqdm(range(epoch)):
        # 选择优化器和学习率调度器
        optimizer_1, _, scheduler = get_optimizer(args, model)
        # 如果需要训练
        if need_train:
            train_loss, val_loss, val_acc, train_acc = train(model, data, optimizer_1, loss_fn, train_mask, val_mask, args.no_val, llm_pred, conf)
            # 如果启用了学习率调度器，则进行学习率调度
            if scheduler:
                scheduler.step()
            # 如果启用了输出中间结果并且不禁用验证集，则打印训练损失、验证损失和验证准确率。
            # if args.output_intermediate and not args.no_val:
            #     print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
            # print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc}")
            test_acc, res = test(model, data, 0, train_mask)
            precisions_1.append(test_acc)
            test_acc, res = test(model, data, 0, test_mask)
            precisions_2.append(test_acc)
            test_acc, res = test(model, data, 0, data.test_mask)
            precisions_3.append(test_acc)
            test_acc, res = test(model, data, 0, val_mask)
            precisions_4.append(test_acc)
            losses.append(val_loss)
            #  如果启用了调试模式，则计算并保存训练集和测试集的准确率
            if args.debug:
                if args.filter_strategy == 'none':
                    test_acc, res = test(model, data, 0, test_mask)
                else:
                    test_acc, res = test(model, data, 0, test_mask, data.backup_y)
                # print(f"Epoch {i}: Test acc: {test_acc}")
                debug_acc.append(test_acc)
                this_train_acc.append(train_acc)
            # 如果不禁用验证集，并且当前验证准确率超过之前的最佳验证准确率，则保存当前模型，并重置早停计数器
            if not args.no_val:
                if val_acc > best_val:
                    best_val = val_acc
                    best_model = deepcopy(model)
                    # print(f"+++++++++++++++++++++++++++++++epoch: {i}+++++++++++++++++++++++++++++++++++++++++")
                    early_stop_accum = 0
                else:
                    if i >= args.early_stop_start:
                        early_stop_accum += 1
                    # 如果超过阈值则提前停止训练
                    if early_stop_accum > args.early_stopping and i >= args.early_stop_start:
                        print(f"Early stopping at epoch {i}")
                        break
        else:
            best_model = model
    # plot_metrics(precisions_1, precisions_2, precisions_3, precisions_4, args)
    # 基于标签的数据选择则将数据标签恢复为原始标签
    if 'pl' in args.split or 'active' in args.split:
        data.y = data.backup_y
    # 如果禁用了验证集或者没有找到最佳模型,则将当前模型作为最佳模型
    if args.no_val or best_model == None:
        best_model = model
    best_val = 0
    res_ttt = test(best_model, data, 0, data.test_mask, data.y)[0]
    print("round 1 acc not changed:", res_ttt)
    # model_new = get_new_model(best_model, changed_model, fixed_model, args.changed, args.num_layers)
    model_new = deepcopy(best_model)
    res_ttt = test(model_new, data, 0, data.test_mask, data.y)[0]
    print("round 1 acc:", res_ttt)
    print("========================== continue training ==========================")
    loss_fn = None
    precisions_1 = []
    precisions_2 = []
    precisions_3 = []
    precisions_4 = []
    losses = []
    test_acc, res = test(model_new, data, 0, train_mask)
    precisions_1.append(test_acc)
    test_acc, res = test(model_new, data, 0, test_mask)
    precisions_2.append(test_acc)
    test_acc, res = test(model_new, data, 0, data.test_mask)
    precisions_3.append(test_acc)
    test_acc, res = test(model_new, data, 0, val_mask)
    precisions_4.append(test_acc)
    gcn_info = get_info_GCN(model_new, data, args)
    conf_gcn = gcn_info['confidence']
    for i in tqdm(range(epoch)):
        _, optimizer_2, scheduler = get_optimizer(args, model_new)
        # 如果需要训练
        if need_train:
            train_loss, val_loss, val_acc, train_acc = c_train(model_new, data, optimizer_2, loss_fn, train_mask, val_mask, args.no_val, llm_pred, conf_gcn)
            # 如果启用了学习率调度器，则进行学习率调度
            if scheduler:
                scheduler.step()
            # 如果启用了输出中间结果并且不禁用验证集，则打印训练损失、验证损失和验证准确率。
            # if args.output_intermediate and not args.no_val:
            #     print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc[0]}")
            # print(f"Epoch {i}: Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc}")
            test_acc, res = test(model_new, data, 0, train_mask)
            precisions_1.append(test_acc)
            test_acc, res = test(model_new, data, 0, test_mask)
            precisions_2.append(test_acc)
            test_acc, res = test(model_new, data, 0, data.test_mask)
            precisions_3.append(test_acc)
            test_acc, res = test(model_new, data, 0, val_mask)
            precisions_4.append(test_acc)
            losses.append(val_loss)
            #  如果启用了调试模式，则计算并保存训练集和测试集的准确率
            if args.debug:
                if args.filter_strategy == 'none':
                    test_acc, res = test(model_new, data, 0, test_mask)
                else:
                    test_acc, res = test(model_new, data, 0, test_mask, data.backup_y)
                # print(f"Epoch {i}: Test acc: {test_acc}")
                debug_acc.append(test_acc)
                this_train_acc.append(train_acc)
            # 如果不禁用验证集，并且当前验证准确率超过之前的最佳验证准确率，则保存当前模型，并重置早停计数器
            if not args.no_val:
                if val_acc > best_val:
                    best_val = val_acc
                    best_model = deepcopy(model_new)
                    # print(f"+++++++++++++++++++++++++++++++epoch: {i}+++++++++++++++++++++++++++++++++++++++++")
                    early_stop_accum = 0
                else:
                    if i >= args.early_stop_start:
                        early_stop_accum += 1
                    # 如果超过阈值则提前停止训练
                    if early_stop_accum > args.early_stopping and i >= args.early_stop_start:
                        print(f"Early stopping at epoch {i}")
                        break
        else:
            best_model = model_new
    # plot_metrics(precisions_1, precisions_2, precisions_3, precisions_4, args)
    res_ttt = test(best_model, data, 0, data.test_mask, data.y)[0]
    print("round 2 acc:", res_ttt)
    # model_new = get_new_model(best_model, changed_model, fixed_model, args.changed, args.num_layers)
    return best_model


def main(data_path, args = None, custom_args = None, save_best = False):
    seeds = [i for i in range(args.main_seed_num)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_ttt = []
    results_org = []
    print('===========')
    reset_args(args)
    print("args in this loop：", args)
    degree_concept, degree_covariate, time_concept, time_covariate = load_ood(args.dataset, data_path, 'GOOD')
    data_all = time_concept
    vars(args)['input_dim'] = data_all.x.shape[1]
    vars(args)['num_classes'] = data_all.y.max().item() + 1
    data_all.x = data_all.x.to(torch.float32)
    data_all.y = data_all.y.to(torch.long)
    data_all = data_all.to(device)
    for seed in seeds:
        model = pre_train(data_all, seed, args)
        # model = get_model(args).to(device)
        res_train = test(model, data_all, 0, data_all.train_mask, data_all.y)[0]
        print("train acc:", res_train)
        res_org = test(model, data_all, 0, data_all.test_mask, data_all.y)[0]
        results_org.append(res_org)
        print("pretrain GCN on test sampels:", res_org)
    # select_ids = idxs[select_samples]
    test_index = torch.nonzero(data_all.test_mask).squeeze()
    pl_data_path = osp.join(data_path, "active", f"{args.dataset}^cache^{args.filter_strategy}.pt")
    pl_data = torch.load(pl_data_path, map_location='cpu')
    llm_pred = pl_data['pred']
    conf = pl_data['conf']
    # conf = torch.Tensor([conf[i].item() for i in test_index])
    # TODO: one function to get select_masks decided by budget
    # llm_pred_test = torch.Tensor([llm_pred[i].item() for i in test_index]).long()
    budget = args.total_budget
    filename = f"D:/code/LLM_data/concept_word/{args.dataset}_{args.strategy}_{args.second_filter}_b{budget}.pt"
    # filename = f"/home/zjx/code/LLM_data/concept_word/{args.dataset}_{args.strategy}_{args.second_filter}_b{budget}.pt"
    test_mask = data_all.test_mask.cpu()
    if os.path.isfile(filename):
        data_all = torch.load(filename)
    else:
        test_data = extract_subgraph(data_all, data_all.test_mask)
        reliability_list = []
        gcn_info = get_info_GCN(model, test_data, args)
        train_masks, val_masks, test_masks = active_selection(args, test_data, llm_pred[test_mask], conf[test_mask], args.split,
                                                           gcn_info, budget, args.strategy,
                                                           args.num_centers,
                                                           args.compensation, args.filter_strategy,
                                                           args.max_part, args.oracle, reliability_list,
                                                           args.second_filter, True,
                                                           args.filter_all_wrong_labels, args.alpha,
                                                           args.beta, args.gamma,
                                                           args.ratio)
        # test_data.train_masks = train_masks
        # test_data.val_masks = val_masks
        # test_data.test_masks = test_masks
        # torch.save(test_data, filename)
        for i, test_train in enumerate(train_masks):
            m = 0
            mask = torch.Tensor([False for i in range(data_all.x.shape[0])]).bool()
            for j in range(data_all.x.shape[0]):
                if data_all.test_mask[j]:
                    if test_train[m]:
                        mask[j] = True
                    m += 1
            train_masks[i] = mask

        for i, test_test in enumerate(test_masks):
            m = 0
            mask = torch.Tensor([False for i in range(data_all.x.shape[0])]).bool()
            for j in range(data_all.x.shape[0]):
                if data_all.test_mask[j]:
                    if test_test[m]:
                        mask[j] = True
                    m += 1
            test_masks[i] = mask

        for i, test_val in enumerate(val_masks):
            m = 0
            mask = torch.Tensor([False for i in range(data_all.x.shape[0])]).bool()
            for j in range(data_all.x.shape[0]):
                if data_all.test_mask[j]:
                    if test_val[m]:
                        mask[j] = True
                    m += 1
            val_masks[i] = mask
        data_all.train_masks = train_masks
        data_all.val_masks = val_masks
        data_all.test_masks = test_masks
        torch.save(data_all, filename)
    print("Selection completed.")
    for i, seed in enumerate(seeds):
        seed_everything(seed)
        # select_samples = select_masks[seed]
        # llm_pred[select_samples == False] = -1
        model_new = test_time_training(model, seed, args, data_all, llm_pred, conf)
        res_ttt = test(model_new, data_all, 0, data_all.test_mask, data_all.y)[0]
        results_ttt.append(res_ttt)
    return np.var(results_org), np.mean(results_ttt)

def sweep(trial):
    # args.num_layers = trial.suggest_int("num_layers", 1, 6, 1, log=False)
    # args.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    # args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    # args.dropout = trial.suggest_float("dropout", 0, 0.5, log=False)
    reset_args(args)
    # args.changed = trial.suggest_int("changed", 1, args.num_layers, 1, log=False)
    args.lr_ttt_2 = trial.suggest_float("lr_ttt_2", 1e-5, 1e-2, log=True)
    args.lr_ttt_1 = trial.suggest_float("lr_ttt_1", 1e-5, 1e-2, log=True)
    args.epochs = 50
    seeds = [i for i in range(args.main_seed_num)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_ttt = []
    results_org = []
    print('===========')
    print("args in this loop：", args)
    degree_concept, degree_covariate, time_concept, time_covariate = load_ood(args.dataset, data_path, 'GOOD')
    data_all = time_concept
    vars(args)['input_dim'] = data_all.x.shape[1]
    vars(args)['num_classes'] = data_all.y.max().item() + 1
    data_all.x = data_all.x.to(torch.float32)
    data_all.y = data_all.y.to(torch.long)
    data_all = data_all.to(device)
    for seed in seeds:
        model = pre_train(data_all, seed, args)
        # model = get_model(args).to(device)
        res_train = test(model, data_all, 0, data_all.train_mask, data_all.y)[0]
        print("train acc:", res_train)
        res_org = test(model, data_all, 0, data_all.test_mask, data_all.y)[0]
        results_org.append(res_org)
        print("pretrain GCN on test sampels:", res_org)
    # select_ids = idxs[select_samples]
    test_index = torch.nonzero(data_all.test_mask).squeeze()
        #llm_pred = load_llm_pred_me(args.dataset).to(device)
    pl_data_path = osp.join(data_path, "active", f"{args.dataset}^cache^{args.filter_strategy}.pt")
    pl_data = torch.load(pl_data_path, map_location='cpu')
    llm_pred = pl_data['pred']
    conf = pl_data['conf']
    # conf = torch.Tensor([conf[i].item() for i in test_index])
    # TODO: one function to get select_masks decided by budget
    # llm_pred = torch.Tensor([llm_pred[i].item() for i in test_index]).long()
    budget = args.total_budget
    filename = f"D:/code/LLM_data/concept_word/{args.dataset}_{args.strategy}_{args.second_filter}_b{budget}.pt"
    # filename = f"/home/zjx/code/LLM_data/concept_word/{args.dataset}_{args.strategy}_{args.second_filter}_b{budget}.pt"
    test_mask = data_all.test_mask.cpu()
    if os.path.isfile(filename):
        data_all = torch.load(filename)
    else:
        test_data = extract_subgraph(data_all, data_all.test_mask)
        reliability_list = []
        gcn_info = get_info_GCN(model, test_data, args)
        train_masks, val_masks, test_masks = active_selection(args, test_data, llm_pred[test_mask], conf, args.split,
                                                              gcn_info, budget, args.strategy,
                                                              args.num_centers,
                                                              args.compensation, args.filter_strategy,
                                                              args.max_part, args.oracle, reliability_list,
                                                              args.second_filter, True,
                                                              args.filter_all_wrong_labels, args.alpha,
                                                              args.beta, args.gamma,
                                                              args.ratio)
        # test_data.train_masks = train_masks
        # test_data.val_masks = val_masks
        # test_data.test_masks = test_masks
        # torch.save(test_data, filename)
        for i, test_train in enumerate(train_masks):
            m = 0
            mask = torch.Tensor([False for i in range(data_all.x.shape[0])]).bool()
            for j in range(data_all.x.shape[0]):
                if data_all.test_mask[j]:
                    if test_train[m]:
                        mask[j] = True
                    m += 1
            train_masks[i] = mask

        for i, test_test in enumerate(test_masks):
            m = 0
            mask = torch.Tensor([False for i in range(data_all.x.shape[0])]).bool()
            for j in range(data_all.x.shape[0]):
                if data_all.test_mask[j]:
                    if test_test[m]:
                        mask[j] = True
                    m += 1
            test_masks[i] = mask

        for i, test_val in enumerate(val_masks):
            m = 0
            mask = torch.Tensor([False for i in range(data_all.x.shape[0])]).bool()
            for j in range(data_all.x.shape[0]):
                if data_all.test_mask[j]:
                    if test_val[m]:
                        mask[j] = True
                    m += 1
            val_masks[i] = mask
        data_all.train_masks = train_masks
        data_all.val_masks = val_masks
        data_all.test_masks = test_masks
        torch.save(data_all, filename)
    print("Selection completed.")
    for i, seed in enumerate(seeds):
        seed_everything(seed)
        # select_samples = select_masks[seed]
        # llm_pred[select_samples == False] = -1
        model_new = test_time_training(model, seed, args, data_all, llm_pred, conf)
        res_ttt = test(model_new, data_all, 0, data_all.test_mask, data_all.y)[0]
        results_ttt.append(res_ttt)
    return np.mean(results_ttt)

if __name__ == '__main__':
    current_time = int(time.time())
    print("Start")
    args = get_command_line_args()
    params_dict = load_yaml(args.yaml_path)
    data_path = params_dict['DATA_PATH']
    datasets = ['cora', 'pubmed', 'citeseer', 'wikics', 'arxiv']
    #datasets = ['arxiv']
    if args.mode == "main":
        for db in datasets:
            if db == 'cora':
                budgets = [77]
            elif db == 'pubmed':
                budgets = [619]
            elif db == 'citeseer':
                budgets = [80]
            elif db == 'wikics':
                budgets = [339]
            elif db == 'arxiv':
                budgets = [4791]
            for b in budgets:
                print("budget: %s; database: %s" % (b, db))
                args.total_budget = b
                args.dataset = db
                bb = args.model_name
                org, ttt = main(data_path, args = args)
                print("result_org: %s; result_ttt: %s" % (org, ttt))
                # f = open('result.txt', 'a')
                # f.write("(database: %s backbone: %s ): result_org: %s; result_ttt: %s\n" % (db, bb, org, ttt))
                # f.close()
    else:
        for db in datasets:
            args.dataset = db
            if db == 'cora':
                budgets = [77]
            elif db == 'pubmed':
                budgets = [619]
            elif db == 'citeseer':
                budgets = [80]
            elif db == 'wikics':
                budgets = [339]
            elif db == 'arxiv':
                budgets = [4791]
            for b in budgets:
                args.total_budget = b
                storage_name = "sqlite:///%s_%s_active_withPS.db" % (db, b)
                study = optuna.create_study(
                    study_name="LLM-TTT-Study",
                    #storage=storage_name,
                    load_if_exists=True,
                    direction="maximize"
                )
                study.optimize(sweep, n_trials=500)
                #df = study.trials_dataframe(attrs =('number', 'value', 'params', 'state'))
                df = study.trials_dataframe()
                filename = f"D:/code/LLM/LLMTTT/some TTT results/all_data/concept_word/{db}_{b}_{args.strategy}_{args.second_filter}_{args.filter_strategy}_no_changed.csv"
                # filename = f"/home/zjx/code/LLMTTT/some TTT results/all_data/concept_word/{db}_{b}_{args.strategy}_{args.second_filter}_{args.filter_strategy}_no_changed.csv"
                df.to_csv(filename, index=False)
                df = pd.read_csv(filename)
                df.to_excel(f"D:/code/LLM/LLMTTT/all_data/concept_word/{db}_{b}_{args.strategy}_{args.second_filter}_{args.filter_strategy}_no_changed.xlsx", index=False)
                # df.to_excel(f"/home/code/zjx/LLMTTT/all_data/concept_word/{db}_{b}_{args.strategy}_{args.second_filter}_{args.filter_strategy}_no_changed.xlsx", index=False)
                print(df)
            print("Mode not supported")
        #sweep(data_path, args = args)