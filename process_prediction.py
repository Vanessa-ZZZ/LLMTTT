import json
import torch
import os.path as osp
def load_ood(dataset, data_path, method):
    if method == "GOOD":
        data_path = data_path + "/GOOD"
        if dataset == "arxiv":
            degree_concept = torch.load(osp.join(data_path, "GOODArxiv/degree/processed/concept.pt"))
            degree_concept = degree_concept[0]
            degree_covariate = torch.load(osp.join(data_path, "GOODArxiv/degree/processed/covariate.pt"))
            degree_covariate = degree_covariate[0]
            time_concept = torch.load(osp.join(data_path, "GOODArxiv/time/processed/concept.pt"))
            time_concept = time_concept[0]
            time_covariate = torch.load(osp.join(data_path, "GOODArxiv/time/processed/covariate.pt"))
            time_covariate = time_covariate[0]
            return degree_concept, degree_covariate, time_concept, time_covariate
        else:
            if dataset == "cora":
                data_path = data_path + "/GOODCora"
            elif dataset == "citeseer":
                data_path = data_path + "/GOODCiteseer"
            elif dataset == "pubmed":
                data_path = data_path + "/GOODPubmed"
            elif dataset == "wikics":
                data_path = data_path + "/GOODWikics"
            elif dataset == "products":
                data_path = data_path + "/GOODProducts"
            else:
                raise NotImplementedError
            degree_concept = torch.load(osp.join(data_path, "degree/processed/concept.pt"))
            degree_concept = degree_concept[0]
            degree_covariate = torch.load(osp.join(data_path, "degree/processed/covariate.pt"))
            degree_covariate = degree_covariate[0]
            word_concept = torch.load(osp.join(data_path, "word/processed/concept.pt"))
            word_concept = word_concept[0]
            word_covariate = torch.load(osp.join(data_path, "word/processed/covariate.pt"))
            word_covariate = word_covariate[0]
            return degree_concept, degree_covariate, word_concept, word_covariate
    elif method == "EERM":
        data_path = data_path + "/EERM"
        train_data = torch.load(osp.join(data_path, f"{dataset}_ood_train_gcn.pt"), map_location='cpu')
        val_data = torch.load(osp.join(data_path, f"{dataset}_ood_val_gcn.pt"), map_location='cpu')
        test_datas = []
        for i in range(5):
            test_data = torch.load(osp.join(data_path, f"{dataset}_ood_test_gcn_{i + 1}.pt"), map_location='cpu')
            test_datas.append(test_data)
        return train_data, val_data, test_datas
    else:
        raise NotImplementedError
data_path = "D:/code/LLM_data"
for dataset in ['cora', 'pubmed', 'citeseer', 'wikics', 'arxiv']:
    degree_concept, degree_covariate, time_concept, time_covariate = load_ood(dataset, data_path, 'GOOD')
    data_all = [degree_concept, degree_covariate, time_concept, time_covariate]
    for data in data_all:
        print("%s : %s" % (dataset, data.test_mask.sum()))