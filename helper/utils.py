import random, os
import numpy as np
import torch
from torch_geometric.utils import to_undirected
import sys
import torch.nn as nn
import yaml
import seaborn as sns
import ast
import editdistance
from collections import Counter
import matplotlib.pyplot as plt
from ogb.nodeproppred import Evaluator
from copy import deepcopy
import itertools
import torch.nn.functional as F
import pandas as pd

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_dict = yaml.safe_load(file)
            return yaml_dict
        except yaml.YAMLError as e:
            print(f"Error while parsing YAML file: {e}")
def soft_selection(train_mask, training_score, num_select):
    total_score = sum(training_score)
    probabilities = [(score / total_score).item() for score in training_score]
    #   probabilities = probabilities.numpy()  # 转换为张量
    # probabilities /= torch.sum(probabilities)  # 将概率归一化到总和为 1
    # while(sum(probabilities) != 1):
    #     total_sum = sum(probabilities)
    #     probabilities = [score / total_sum for score in probabilities]
    # probabilities = np.array(probabilities)
    # probabilities = np.random.normalize(probabilities, norm='l1')
    probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
    selected_idx = np.random.choice(len(training_score), num_select, replace=False, p=probabilities)
    selected_idx = torch.tensor(selected_idx)
    train_mask[selected_idx] = True
    return train_mask

def retrieve_dict(clean_t):
    start = clean_t.find('[')
    end = clean_t.find(']', start) + 1  # +1 to include the closing bracket
    list_str = clean_t[start:end]
    result = ast.literal_eval(list_str)
    return result


def plot_acc_calibration(idx_test, output, labels, n_bins, title):
    output = torch.softmax(output, dim=1)
    pred_label = torch.max(output[idx_test], 1)[1]
    p_value = torch.max(output[idx_test], 1)[0]
    ground_truth = labels[idx_test]
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
    for index, value in enumerate(p_value):
        #value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) -0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]

    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    plt.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
            alpha=0.7, width=0.03, color='dodgerblue', label='Outputs')
    plt.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.03, color='lightcoral', label='Expected')
    plt.plot([0,1], [0,1], ls='--',c='k')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    #title = 'Uncal. - Cora - 20 - GCN'
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.savefig('output/' + title +'.png', format='png', dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    plt.show()

def plot_histograms(content_a, content_b, title, labeltitle, n_bins=50, norm_hist=True):
    # Plot histogram of correctly classified and misclassified examples
    global conf_histogram

    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    sns.distplot(content_a, kde=False, bins=n_bins, norm_hist=False, fit=None, label=labeltitle[0])
    sns.distplot(content_b, kde=False, bins=n_bins, norm_hist=False,  fit=None, label=labeltitle[1])
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.savefig('output/' + title +'.png', format='png', transparent=True, dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    plt.show()

def get_info_GCN(model, test_data, args):
    model.eval()
    with torch.no_grad():
        logits = model(test_data)
        pred = logits.argmax(dim=-1, keepdim=True)
        softmax = torch.nn.Softmax(dim=1)
        prob = softmax(logits)
        entropy = -torch.sum(prob * torch.log2(prob), dim=1)
        confidence = []
        for i in range(test_data.x.shape[0]):
            confidence.append(prob[i][pred[i]].item())
    info_dict = {
        'confidence': torch.tensor(confidence),
        'entropy': torch.tensor(entropy.tolist())
    }
    return info_dict

def get_closest_label(input_string, label_names):
    min_distance = float('inf')
    closest_label = None

    for label in label_names:
        # editdistance.eval 函数计算编辑距离（编辑距离表示将一个字符串转换为另一个字符串所需的最少编辑操作次数，例如插入、删除和替换字符）
        distance = editdistance.eval(input_string, label)
        if distance < min_distance:
            min_distance = distance
            closest_label = label

    return closest_label


import torch


def soft_match_weighting(logits_x_ulb, num_classes, n_sigma=2, momentum=0.999, per_class=False):
    """
    SoftMatch learnable truncated Gaussian weighting

    Args:
        logits_x_ulb: tensor, shape [batch_size, num_classes], the output logits from the model.
        num_classes: int, the number of classes.
        n_sigma: int, the standard deviation multiplier.
        momentum: float, the momentum coefficient for the moving average.
        per_class: bool, whether to compute the mean and variance for each class separately.

    Returns:
        mask: tensor, shape [batch_size], the computed weight for each sample.
    """
    prob_max_mu_t = torch.ones((num_classes)) / num_classes if per_class else (1.0 / num_classes)
    prob_max_var_t = torch.ones((num_classes)) if per_class else 1.0
    m = momentum
    n_sigma = n_sigma
    max_probs, max_idx = torch.max(torch.softmax(logits_x_ulb.detach(), dim=-1), dim=-1)
    if per_class:
        prob_max_mu_t = torch.zeros_like(prob_max_mu_t)
        prob_max_var_t = torch.ones_like(prob_max_var_t)
        for i in range(num_classes):
            prob = max_probs[max_idx == i]
            if len(prob) > 1:
                prob_max_mu_t[i] = torch.mean(prob)
                prob_max_var_t[i] = torch.var(prob, unbiased=True)
    else:
        prob_max_mu_t = torch.mean(max_probs)  # torch.quantile(max_probs, 0.5)
        prob_max_var_t = torch.var(max_probs, unbiased=True)
    if not prob_max_mu_t.is_cuda:
        prob_max_mu_t = prob_max_mu_t.to(logits_x_ulb.device)
    if not prob_max_var_t.is_cuda:
        prob_max_var_t = prob_max_var_t.to(logits_x_ulb.device)
    mu = prob_max_mu_t[max_idx] if per_class else prob_max_mu_t
    var = prob_max_var_t[max_idx] if per_class else prob_max_var_t
    mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (n_sigma ** 2))))
    return mask


# @torch.no_grad()
# def masking(node_confidence, n_sigma=2):
#     prob_max_mu_t = torch.mean(node_confidence)
#     prob_max_var_t = torch.var(node_confidence, unbiased=True)
#
#     mask = torch.exp(
#         -((torch.clamp(node_confidence - prob_max_mu_t, max=0.0) ** 2) / (2 * prob_max_var_t / (n_sigma ** 2))))
#
#     return mask
def ce_loss(logits, targets, reduction='none'):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def forward(self, logits, targets, reduction='none'):
        return ce_loss(logits, targets, reduction)

def consistency_loss(logits, targets, name='ce', mask=None):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()



class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits, targets, name, mask)
def plot_metrics(precisions_1, precisions_2, precisions_3, precisions_4, args):
    """
    训练指标变化过程可视化
    :param precisions:
    :param recalls:
    :param f1s:
    :param losses:
    :return:
    """
    epochs = range(1, len(precisions_1) + 1)
    plt.figure(figsize=(10, 8))
    plt.plot(epochs, precisions_1, 'g', label='train')
    plt.plot(epochs, precisions_2, 'b', label='test')
    plt.plot(epochs, precisions_3, 'r', label='all')
    plt.plot(epochs, precisions_4, 'y', label='val')
    #plt.plot(epochs, losses, 'b', label='Loss')
    plt.title('TTT from pretrain model (my TTT)')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    filename = f"D:\code\LLM\LLMTTT\some exp results/{args.dataset}_{args.probability}.png"
    #plt.savefig(filename)
    plt.show()
# def plot_metrics(precisions, losses):
#     """
#     训练指标变化过程可视化
#     :param precisions:
#     :param recalls:
#     :param f1s:
#     :param losses:
#     :return:
#     """
#     epochs = range(1, len(precisions) + 1)
#     plt.figure(figsize=(10, 8))
#     plt.plot(epochs, precisions, 'g', label='Precision')
#     #plt.plot(epochs, losses, 'b', label='Loss')
#     plt.title('Training And Validation Metrics')
#     plt.xlabel('Epochs')
#     plt.ylabel('Metrics')
#     plt.legend()
#     plt.show()

def eval_val(y, pred, mask):
    pred = pred.argmax(dim=-1, keepdim=True)
    if len(y.shape) == 1:
        y = y.unsqueeze(dim=1)
    else:
        y = y
    evaluator = Evaluator(name='ogbn-arxiv')
    acc = evaluator.eval({
        'y_true': y[mask],
        'y_pred': pred[mask],
    })['acc']
    return acc
def most_common_number(numbers):
    counter = Counter(numbers)
    most_common = counter.most_common(1)  # Get the most common number and its count
    return most_common[0][0]
def retrieve_answer(answer, label_names):
    output = []
    invalid = 0
    for result in answer:
        # import ipdb; ipdb.set_trace()
        if result == "":
            res = [("", 0)]
            output.append(res)
            continue
        line = result[0].lower()
        try:
            this_dict = retrieve_dict(line)
            res = []
            for dic in this_dict:
                answer = dic['answer']
                confidence = dic['confidence']
                if isinstance(confidence, str) and '%' in confidence:
                    confidence = int(confidence.replace('%', ''))
                res.append((answer, confidence))
            output.append(res)
        except:
            answer = get_closest_label(line, label_names)
            confidence = min(int(''.join(filter(str.isdigit, line))), 100)
            # res = [("", 0)]
            res = [(answer, confidence)]
            output.append(res)
            invalid += 1
            continue
    print("invalid number: {}".format(invalid))
    return output

def load_mapping():
    arxiv_mapping = {
    'arxiv cs ai': 'cs.AI',
    'arxiv cs cl': 'cs.CL',
    'arxiv cs cc': 'cs.CC',
    'arxiv cs ce': 'cs.CE',
    'arxiv cs cg': 'cs.CG',
    'arxiv cs gt': 'cs.GT',
    'arxiv cs cv': 'cs.CV',
    'arxiv cs cy': 'cs.CY',
    'arxiv cs cr': 'cs.CR',
    'arxiv cs ds': 'cs.DS',
    'arxiv cs db': 'cs.DB',
    'arxiv cs dl': 'cs.DL',
    'arxiv cs dm': 'cs.DM',
    'arxiv cs dc': 'cs.DC',
    'arxiv cs et': 'cs.ET',
    'arxiv cs fl': 'cs.FL',
    'arxiv cs gl': 'cs.GL',
    'arxiv cs gr': 'cs.GR',
    'arxiv cs ar': 'cs.AR',
    'arxiv cs hc': 'cs.HC',
    'arxiv cs ir': 'cs.IR',
    'arxiv cs it': 'cs.IT',
    'arxiv cs lo': 'cs.LO',
    'arxiv cs lg': 'cs.LG',
    'arxiv cs ms': 'cs.MS',
    'arxiv cs ma': 'cs.MA',
    'arxiv cs mm': 'cs.MM',
    'arxiv cs ni': 'cs.NI',
    'arxiv cs ne': 'cs.NE',
    'arxiv cs na': 'cs.NA',
    'arxiv cs os': 'cs.OS',
    'arxiv cs oh': 'cs.OH',
    'arxiv cs pf': 'cs.PF',
    'arxiv cs pl': 'cs.PL',
    'arxiv cs ro': 'cs.RO',
    'arxiv cs si': 'cs.SI',
    'arxiv cs se': 'cs.SE',
    'arxiv cs sd': 'cs.SD',
    'arxiv cs sc': 'cs.SC',
    'arxiv cs sy': 'cs.SY'
    }

    # arxiv_mapping = {'arxiv cs ai': 'Artificial Intelligence', 'arxiv cs cl': 'Computation and Language', 'arxiv cs cc': 'Computational Complexity', 'arxiv cs ce': 'Computational Engineering, Finance, and Science', 'arxiv cs cg': 'Computational Geometry', 'arxiv cs gt': 'Computer Science and Game Theory', 'arxiv cs cv': 'Computer Vision and Pattern Recognition', 'arxiv cs cy': 'Computers and Society', 'arxiv cs cr': 'Cryptography and Security', 'arxiv cs ds': 'Data Structures and Algorithms', 'arxiv cs db': 'Databases', 'arxiv cs dl': 'Digital Libraries', 'arxiv cs dm': 'Discrete Mathematics', 'arxiv cs dc': 'Distributed, Parallel, and Cluster Computing', 'arxiv cs et': 'Emerging Technologies', 'arxiv cs fl': 'Formal Languages and Automata Theory', 'arxiv cs gl': 'General Literature', 'arxiv cs gr': 'Graphics', 'arxiv cs ar': 'Hardware Architecture', 'arxiv cs hc': 'Human-Computer Interaction', 'arxiv cs ir': 'Information Retrieval', 'arxiv cs it': 'Information Theory', 'arxiv cs lo': 'Logic in Computer Science', 'arxiv cs lg': 'Machine Learning', 'arxiv cs ms': 'Mathematical Software', 'arxiv cs ma': 'Multiagent Systems', 'arxiv cs mm': 'Multimedia', 'arxiv cs ni': 'Networking and Internet Architecture', 'arxiv cs ne': 'Neural and Evolutionary Computing', 'arxiv cs na': 'Numerical Analysis', 'arxiv cs os': 'Operating Systems', 'arxiv cs oh': 'Other Computer Science', 'arxiv cs pf': 'Performance', 'arxiv cs pl': 'Programming Languages', 'arxiv cs ro': 'Robotics', 'arxiv cs si': 'Social and Information Networks', 'arxiv cs se': 'Software Engineering', 'arxiv cs sd': 'Sound', 'arxiv cs sc': 'Symbolic Computation', 'arxiv cs sy': 'Systems and Control'}
    citeseer_mapping = {
        "Agents": "Agents",
        "ML": "Machine Learning",
        "IR": "Information Retrieval",
        "DB": "Database",
        "HCI": "Human Computer Interaction",
        "AI": "Artificial Intelligence"
    }
    pubmed_mapping = {
        'Diabetes Mellitus, Experimental': 'Diabetes Mellitus Experimental',
        'Diabetes Mellitus Type 1': 'Diabetes Mellitus Type 1',
        'Diabetes Mellitus Type 2': 'Diabetes Mellitus Type 2'
    }
    cora_mapping = {
        'Rule_Learning': "Rule_Learning",
        'Neural_Networks': "Neural_Networks",
        'Case_Based': "Case_Based",
        'Genetic_Algorithms': "Genetic_Algorithms",
        'Theory': "Theory",
        'Reinforcement_Learning': "Reinforcement_Learning",
        'Probabilistic_Methods': "Probabilistic_Methods"
    }

    products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}

    wikics_mapping = {
        'Computational linguistics': 'Computational linguistics',
        'Databases': 'Databases',
        'Operating systems': 'Operating systems',
        'Computer architecture': 'Computer architecture',
        'Computer security': 'Computer security',
        'Internet protocols': 'Internet protocols',
        'Computer file systems': 'Computer file systems',
        'Distributed computing architecture': 'Distributed computing architecture',
        'Web technology': 'Web technology',
        'Programming language topics': 'Programming language topics'
    }

    tolokers_mapping = {
        'not banned': 'not banned',
        'banned': 'banned'
    }

    twenty_newsgroup_mapping = {'alt.atheism': 'News about atheism.', 'comp.graphics': 'News about computer graphics.', 'comp.os.ms-windows.misc': 'News about Microsoft Windows.', 'comp.sys.ibm.pc.hardware': 'News about IBM PC hardware.', 'comp.sys.mac.hardware': 'News about Mac hardware.', 'comp.windows.x': 'News about the X Window System.', 'misc.forsale': 'Items for sale.', 'rec.autos': 'News about automobiles.', 'rec.motorcycles': 'News about motorcycles.', 'rec.sport.baseball': 'News about baseball.', 'rec.sport.hockey': 'News about hockey.', 'sci.crypt': 'News about cryptography.', 'sci.electronics': 'News about electronics.', 'sci.med': 'News about medicine.', 'sci.space': 'News about space and astronomy.', 'soc.religion.christian': 'News about Christianity.', 'talk.politics.guns': 'News about gun politics.', 'talk.politics.mideast': 'News about Middle East politics.', 'talk.politics.misc': 'News about miscellaneous political topics.', 'talk.religion.misc': 'News about miscellaneous religious topics.'}



    return {
        'arxiv': arxiv_mapping,
        'citeseer': citeseer_mapping,
        'pubmed': pubmed_mapping,
        'cora': cora_mapping,
        'products': products_mapping,
        'wikics': wikics_mapping,
        'tolokers': tolokers_mapping,
        '20newsgroup': twenty_newsgroup_mapping
    }

def seed_everything(seed: int):
    #dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    #os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    #torch.use_deterministic_algorithms(True)
def generate_random_mask(n, total_elements):
    mask = [1] * n + [0] * (total_elements - n)
    random.shuffle(mask)
    return mask
def reset_args(args):
    # if args.dataset in ['cora', 'pubmed', 'citeseer']:
    #     args.budget = 940
    # elif args.dataset == 'arxiv':
    #     args.budget = 800
    # elif args.dataset == 'wikics':
    #     args.budget = 800
    # elif args.dataset == 'products':
    #     args.total_budget = 800
    # else:
    #     raise NotImplementedError
    if args.dataset == "cora":
        args.lr = 0.001
        args.weight_decay = 0.001
        args.num_layers = 2
        args.dropout = 0
        if args.strategy == 'degree2':
            args.lr_ttt = 0.003674377
            args.changed = 2
            args.epochs = 56
        elif args.strategy == "pagerank":
            args.lr_ttt = 0.009885528
            args.changed = 1
            args.epochs = 25
        elif args.strategy == 'just_gcn':
            args.lr_ttt_1 = 5.70352E-05
            args.changed = 2
    if args.dataset == "pubmed":
        # args.lr = 0.001
        # args.weight_decay = 0.000132899
        # args.num_layers = 3
        # args.dropout = 0.000132899
        args.lr = 0.01
        args.weight_decay = 0.000132899
        args.num_layers = 3
        args.dropout = 0.5
        if args.strategy == 'degree2':
            args.lr_ttt = 0.009983646
            args.changed = 2
            args.epochs = 9
        elif args.strategy == "pagerank":
            args.lr_ttt = 0.007650958
            args.changed = 2
            args.epochs = 56
        elif args.strategy == 'just_gcn':
            args.lr_ttt_1 = 0.000923257
            args.changed = 3
    if args.dataset == "citeseer":
        args.lr = 0.000996657
        args.weight_decay = 0.000166317
        args.num_layers = 4
        args.dropout = 0.100858804
        if args.strategy == 'degree2':
            args.lr_ttt = 0.000371593
            args.changed = 4
            args.epochs = 58
        elif args.strategy == "pagerank":
            args.lr_ttt = 0.006606836
            args.changed = 4
            args.epochs = 70
        elif args.strategy == 'just_gcn':
            args.lr_ttt_1 = 0.000523479
            args.changed = 4
    if args.dataset == "wikics":
        args.lr = 0.008221264
        args.weight_decay = 0.000721739
        args.num_layers = 2
        args.dropout = 0.277635086
        if args.strategy == 'degree2':
            args.lr_ttt = 0.001355267
            args.changed = 2
            args.epochs = 47
        elif args.strategy == "pagerank":
            args.lr_ttt = 0.009057038
            args.changed = 1
            args.epochs = 33
        elif args.strategy == 'just_gcn':
            args.lr_ttt_1 = 0.000159439
            args.changed = 1
    if args.dataset == "arxiv":
        args.lr = 0.000926368
        args.weight_decay = 7.81E-05
        args.num_layers = 2
        args.dropout = 0.071913764
        if args.strategy == 'degree2':
            args.lr_ttt = 0.003674377
            args.changed = 2
            args.epochs = 56
        elif args.strategy == "pagerank":
            args.lr_ttt = 0.009885528
            args.changed = 1
            args.epochs = 25
    filename = 'pretrain_goodgcn_val/params.xlsx'
    df = pd.read_excel(filename)
    df2 = df[(df.dataset == args.dataset) & (df.model_name == args.model_name)]
    params = df2[['lr', 'num_layers', 'weight_decay', 'dropout']].values
    if len(params) == 1:
        args.lr, args.num_layers, args.weight_decay, args.dropout = params[0]
    args.num_layers = int(args.num_layers)
    # if args.tune == 0:
    #     import pandas as pd
    #     #filename = './models/params.csv'
    #     filename = './params.csv'
    #     df = pd.read_csv(filename, delimiter=',')
    #     df2 = df[(df.dataset == args.dataset) & (df.model == args.model)]
    #     params = df2[['changed', 'lr_ttt', 'epoch']].values
    #     if len(params) == 1:
    #         args.changed, args.lr_ttt, args.epochs = params[0]
    #         args.epochs = int(args.epochs)
from torch_geometric.utils import subgraph
def extract_subgraph(data, mask):
    sub_data = deepcopy(data)
    test_index = torch.nonzero(mask).squeeze()
    edge_index, _ = subgraph(test_index, data.edge_index)
    for i in range(edge_index.shape[1]):
        edge_index[0][i] = torch.where(test_index == edge_index[0][i])[0]
        edge_index[1][i] = torch.where(test_index == edge_index[1][i])[0]
    sub_data.edge_index = edge_index
    sub_data.x = data.x[test_index]
    sub_data.y = data.y[test_index]
    sub_data.train_masks = [data.train_masks[0][test_index]]
    sub_data.val_masks = [data.val_masks[0][test_index]]
    sub_data.test_masks = [data.test_masks[0][test_index]]
    del sub_data.raw_text
    # del sub_data.train_masks
    # del sub_data.val_masks
    # del sub_data.test_masks
    del sub_data.train_mask
    del sub_data.val_mask
    del sub_data.test_mask
    del sub_data.raw_texts
    del sub_data.id_val_mask
    del sub_data.id_test_mask
    return sub_data

def split_data(path, dataset):
    if dataset == 'elliptic':
        path = path + 'temp_elliptic'
        sys.path.append(path)
        from main_as_utils_ell import datasets_tr, datasets_val, datasets_te
        data = [datasets_tr, datasets_val, datasets_te]
    elif dataset == 'fb100':
        path = path + 'multigraph'
        sys.path.append(path)
        from main_as_utils_fb import datasets_tr, datasets_val, datasets_te
        data = [datasets_tr, datasets_val, datasets_te]
    elif dataset == 'amazon-photo':
        path = path + 'synthetic'
        sys.path.append(path)
        from main_as_utils_photo import dataset_tr, dataset_val, datasets_te
        data = [dataset_tr, dataset_val, datasets_te]
    else:
        if dataset == 'cora':
            path = path + 'synthetic'
        elif dataset == 'ogb-arxiv':
            path = path + 'temp_arxiv'
        elif dataset == 'twitch-e':
            path = path + 'multigraph'
        else:
            raise NotImplementedError
        sys.path.append(path)
        print(path)
        from main_as_utils import dataset_tr, dataset_val, datasets_te
        data = [dataset_tr, dataset_val, datasets_te]
    return data
'''
def get_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', name)
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, name)
    elif name in ['arxiv']:
        dataset = PygNodePropPredDataset(name='ogbn-'+name)
    else:
        raise NotImplementedError

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return to_inductive(dataset)
'''
def to_inductive(dataset):
    data = dataset[0]
    add_mask(data, dataset)

    def sub_to_inductive(data, mask):
        new_data = Graph()
        new_data.graph['edge_index'], _ = subgraph(mask, data.edge_index, None, relabel_nodes=True, num_nodes=data.num_nodes)
        new_data.graph['num_nodes'] = mask.sum().item()
        new_data.graph['node_feat'] = data.x[mask]
        new_data.label = data.y[mask].unsqueeze(1)
        return new_data
    train_graph = sub_to_inductive(data, data.train_mask)
    val_graph = sub_to_inductive(data, data.val_mask)
    test_graph = sub_to_inductive(data, data.test_mask)
    val_graph.test_mask = torch.tensor(np.ones(val_graph.graph['num_nodes'])).bool()
    test_graph.test_mask = torch.tensor(np.ones(test_graph.graph['num_nodes'])).bool()
    return [train_graph, val_graph, [test_graph]]

def add_mask(data, dataset):
    # for arxiv
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    n = data.x.shape[0]
    data.train_mask = index_to_mask(train_idx, n)
    data.val_mask = index_to_mask(valid_idx, n)
    data.test_mask = index_to_mask(test_idx, n)
    data.y = data.y.squeeze()
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask
def neighbors(edge_index, node_id):
    row, col = edge_index
    match_idx = torch.where(row == node_id)[0]
    neigh_nodes = col[match_idx]
    return neigh_nodes.tolist()
def get_one_hop_neighbors(data_obj, sampled_test_node_idxs, sample_num = -1):
    ## if sample_nodes == -1, all test nodes within test masks will be considered
    neighbor_dict = {}
    for center_node_idx in sampled_test_node_idxs:
        center_node_idx = center_node_idx.item()
        neighbor_dict[center_node_idx] = neighbors(data_obj.edge_index, center_node_idx)
    return neighbor_dict

def get_two_hop_neighbors_no_multiplication(data_obj, sampled_test_node_idxs, sample_num = -1):
    neighbor_dict = {}
    # for center_node_idx in sampled_test_node_idxs:
    one_hop_neighbor_dict = get_one_hop_neighbors(data_obj, sampled_test_node_idxs)
    for key, value in one_hop_neighbor_dict.items():
        this_key_neigh = []
        second_hop_neighbor_dict = get_one_hop_neighbors(data_obj, torch.IntTensor(value))
        second_hop_neighbors = set(itertools.chain.from_iterable(second_hop_neighbor_dict.values()))
        second_hop_neighbors.discard(key)
        neighbor_dict[key] = sorted(list(second_hop_neighbors))
    return neighbor_dict


def get_sampled_nodes(data_obj, sample_num = -1):
    train_mask = data_obj.train_masks[0]
    # val_mask = data_obj.val_masks[0]
    test_mask = data_obj.test_masks[0]
    all_idxs = torch.arange(data_obj.x.shape[0]).to(data_obj.x.device)
    test_node_idxs = all_idxs[test_mask]
    train_node_idxs = all_idxs[train_mask]
    # val_node_idxs = all_idxs[val_mask]
    if sample_num == -1:
        sampled_test_node_idxs = test_node_idxs
    else:
        sampled_test_node_idxs = test_node_idxs[torch.randperm(test_node_idxs.shape[0])[:sample_num]]
    return sampled_test_node_idxs, train_node_idxs


def query_arxiv_classify_api(title, abstract, url = "http://export.arxiv.org/api/classify"):
    text = title + abstract
    data = {
        "text": text
    }
    r = requests.post(url, data = data)
    return r
