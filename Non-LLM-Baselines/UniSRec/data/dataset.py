import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from recbole.data.dataset import SequentialDataset


class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[int(token)]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding


class FinetuneUniSRecDataset(Dataset):
    def __init__(self, config, gpuid):
        super().__init__()
        self.field2id_token = {'[PAD]':0}
        self.dataset_name = config['dataset']
        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        self.test_data, self.cand_data, plm_embedding_weight = self.load_data(config)
        self.plm_embedding = self.weight2emb(plm_embedding_weight)
        self.device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        seq = self.test_data[index][:-1]
        target = self.test_data[index][-1]
        seq_length = len(seq)
        cand = self.cand_data[index][:-1]
        return seq, seq_length, target, cand

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding

    def load_data(self, config):
        base_path = './dataset/downstream/'
        data_path = base_path + self.dataset_name + '/'
        total_data = np.load(f"{data_path}test_id.npy",allow_pickle=True)
        total_cand_data = np.load(f"{data_path}test_cand_{config['cand_seed']}_id.npy",allow_pickle=True)
        feat_path = osp.join(config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        convert_test_data, convert_cand_data, mapped_feat = self.process_data(total_data, total_cand_data, loaded_feat)
        return convert_test_data, convert_cand_data, mapped_feat

    def process_data(self, test_data, test_cand, plm_embed_weight):
        self.max_seq_length = len(max(test_data, key=len)) - 1
        self.build_fild2id_token(test_data)
        self.build_fild2id_token(test_cand)
        self.item_num = len(self.field2id_token)
        loaded_feat = plm_embed_weight
        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token):
            if token == '[PAD]': continue
            mapped_feat = loaded_feat[int(token)]
        
        convert_test_data = []
        for seq in test_data:
            tmp = self.convert_seq2id(seq)
            convert_test_data.append(tmp)
        
        convert_cand_data = []
        for seq in test_cand:
            tmp = self.convert_seq2id(seq)
            convert_cand_data.append(tmp)

        return convert_test_data, convert_cand_data, mapped_feat

    def build_fild2id_token(self, data):
        for seq in data:
            for item in seq:
                name = str(item)
                if name not in self.field2id_token:
                    self.field2id_token[name] = len(self.field2id_token)
    
    def convert_seq2id(self, seq):
        tmp = []
        for item in seq:
            name = str(item)
            tmp.append(self.field2id_token[name])
        return tmp

class PretrainUniSRecDataset(UniSRecDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_suffix_aug = config['plm_suffix_aug']
        plm_embedding_weight_aug = self.load_plm_embedding(plm_suffix_aug=self.plm_suffix_aug)
        self.plm_embedding_aug = self.weight2emb(plm_embedding_weight_aug)

    def load_plm_embedding(self, plm_suffix_aug=None):
        with open(osp.join(self.config['data_path'], f'{self.dataset_name}.pt_datasets'), 'r') as file:
            dataset_names = file.read().strip().split(',')
        self.dataset_names = dataset_names
        self.logger.info(f'Pre-training datasets: {dataset_names}')

        d2feat = []
        for dataset_name in dataset_names:
            if plm_suffix_aug is None:
                feat_path = osp.join(self.config['data_path'], f'{dataset_name}.{self.plm_suffix}')
            else:
                feat_path = osp.join(self.config['data_path'], f'{dataset_name}.{plm_suffix_aug}')
            loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
            d2feat.append(loaded_feat)

        iid2domain = np.zeros((self.item_num, 1))
        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            did, iid = token.split('-')
            loaded_feat = d2feat[int(did)]
            mapped_feat[i] = loaded_feat[int(iid)]
            iid2domain[i] = int(did)
        self.iid2domain = torch.LongTensor(iid2domain)

        return mapped_feat

class PretrainValDataset(Dataset):
    def __init__(self, config, dataset_names, gpuid) -> None:
        super().__init__()
        self.plm_suffix = config['plm_suffix']
        self.plm_size = config['plm_size']
        self.field2id_token = {'[PAD]':0}
        self.val_data, self.cand_data, plm_embedding_weight = self.load_data(config, dataset_names)
        self.plm_embedding = self.weight2emb(plm_embedding_weight)
        self.device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.val_data)

    def __getitem__(self, index):
        seq = self.val_data[index][:-1]
        target = self.val_data[index][-1]
        seq_length = len(seq)
        cand = self.cand_data[index][:-1]
        return seq, seq_length, target, cand

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding

    def load_data(self, config, dataset_names):
        total_data = []
        total_cand_data = []
        d2feat = []
        base_path = './dataset/pretrain/'
        for dataset_name in dataset_names:
            data_path = base_path + dataset_name + '/'
            total_data.append(np.load(f"{data_path}valid_id.npy",allow_pickle=True))
            total_cand_data.append(np.load(f"{data_path}valid_cand_id.npy",allow_pickle=True))
            feat_path = osp.join(config['data_path'], f'{dataset_name}.{self.plm_suffix}')
            loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
            d2feat.append(loaded_feat)
        
        convert_val_data, convert_cand_data, mapped_feat = self.process_data(total_data, total_cand_data, d2feat)
        return convert_val_data, convert_cand_data, mapped_feat
    
    def process_data(self, val_data, val_cand, plm_embed_weights):
        self.max_seq_length = max(len(max(val_data[0], key=len)), len(max(val_data[1], key=len))) - 1
        for i in range(len(val_data)):
            data = val_data[i]
            self.build_fild2id_token(data, i)
            data = val_cand[i]
            self.build_fild2id_token(data, i)
        self.item_num = len(self.field2id_token)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        for i, token in enumerate(self.field2id_token):
            if token == '[PAD]': continue
            did, iid = token.split('-')
            loaded_feat = plm_embed_weights[int(did)]
            mapped_feat[self.field2id_token[token]] = loaded_feat[int(iid)]
        
        convert_val_data = []
        convert_cand_data = []
        for i in range(len(val_data)):
            data = val_data[i]
            for seq in data:
                tmp = self.convert_seq2id(seq, i)
                convert_val_data.append(tmp)
            data = val_cand[i]
            for seq in data:
                tmp = self.convert_seq2id(seq, i)
                convert_cand_data.append(tmp)
        
        return convert_val_data, convert_cand_data, mapped_feat

    def convert_seq2id(self, seq, idx):
        tmp = []
        for item in seq:
            name = str(idx) + '-' + str(item)
            tmp.append(self.field2id_token[name])
        return tmp

    def build_fild2id_token(self, data, idx):
        for seq in data:
            for item in seq:
                name = str(idx) + '-' + str(item)
                if name not in self.field2id_token:
                    self.field2id_token[name] = len(self.field2id_token)
    