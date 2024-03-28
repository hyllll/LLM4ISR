import argparse
import torch
import os
from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger
from unisrec import UniSRec
from data.dataset import UniSRecDataset, FinetuneUniSRecDataset
from data.dataloader import evaluate_collate_fn
from torch.utils.data import DataLoader
from trainer import FinetuneTrainer
from utils import accuracy_calculator
from config import Best_setting

ACC_KPI = ['ndcg', 'mrr', 'hr']

def finetune(dataset, pretrained_file, cand_seed, fix_enc=True, **kwargs):
    # configurations initialization
    props = ['props/UniSRec.yaml', 'props/finetune.yaml']
    print(props)
    config = Config(model=UniSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)

    config['epochs'] = 100
    config['cand_seed'] = cand_seed
    config['embedding_size'] = Best_setting[dataset]['embedding_size']
    config['train_batch_size'] =  Best_setting[dataset]['train_batch_size']
    config['learning_rate'] = Best_setting[dataset]['learning_rate']
    config['adaptor_layers'] = [config['plm_size'], config['embedding_size']]

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = UniSRecDataset(config)
    train_data, _, _ = data_preparation(config, dataset)
    model = UniSRec(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            logger.info(f'Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False
    logger.info(model)

    trainer = FinetuneTrainer(config, model)
    trainer.fit(
        train_data, saved=True, show_progress=config['show_progress']
    )
    
    # load validation dataset
    test_dataset = FinetuneUniSRecDataset(config, '0')
    test_data = DataLoader(test_dataset, batch_size=128, collate_fn=lambda batch: evaluate_collate_fn(batch, max_length=test_dataset.max_seq_length))
    res_dir = f"res/{config['dataset']}/"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    f = open(res_dir + f"result_{config['dataset']}_{config['cand_seed']}.txt", 'a')
    for k in [1,5,10]:
        line = f'HR@{k}\tNDCG@{k}\tMAP@{k}\n'
        f.write(line)
        preds, truth = trainer.model.cand_sort_predict(test_dataset.plm_embedding, test_data, k=k)
        metrics = accuracy_calculator(preds, truth, ACC_KPI)
        res_line = f'{metrics[2]:.4f}\t{metrics[0]:.4f}\t{metrics[1]:.4f}\n'
        f.write(res_line)
        f.flush()
    f.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='amz', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-f', type=bool, default=True)
    parser.add_argument('-cand_seed', type=int, default=0, help='candidate seed')
    args, unparsed = parser.parse_known_args()
    print(args)

    finetune(args.d, pretrained_file=args.p, cand_seed=args.cand_seed, fix_enc=args.f)