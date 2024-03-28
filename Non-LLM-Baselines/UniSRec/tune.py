import argparse
import optuna
import os
import csv

from logging import getLogger
from recbole.config import Config
from config import HyperParameter_setting
from trainer import TunePretrainTrainer
from recbole.utils import init_seed, init_logger
from torch.utils.data import DataLoader
from utils import accuracy_calculator
from unisrec import UniSRec
from data.dataset import PretrainUniSRecDataset, PretrainValDataset
from data.dataloader import CustomizedTrainDataLoader, evaluate_collate_fn

ACC_KPI = ['ndcg', 'mrr', 'hr']
TRIAL_CNT = 0

def pretrain(args, **kwargs):
    # configurations initialization
    props = ['props/UniSRec.yaml', 'props/pretrain.yaml']
    # print(props)
    # configurations initialization
    config = Config(model=UniSRec, dataset=args.d, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = PretrainUniSRecDataset(config)
    logger.info(dataset)

    pretrain_dataset = dataset.build()[0]
    # pretrain_data = CustomizedTrainDataLoader(config, pretrain_dataset, None, shuffle=True)
    
    # load validation dataset
    dataset_names = pretrain_dataset.dataset_names
    pretrain_val_dataset = PretrainValDataset(config, dataset_names, '0')
    pretrain_val_data = DataLoader(pretrain_val_dataset, batch_size=128, collate_fn=lambda batch: evaluate_collate_fn(batch, max_length=pretrain_val_dataset.max_seq_length))

    tune_params = []
    def objective(trial):
        global TRIAL_CNT
        for key, value in HyperParameter_setting['UniSRec'].items():
            for para_name, scales in value.items():
                config[para_name] = trial.suggest_categorical(para_name, scales)
                tune_params.append(para_name)
        config['adaptor_layers'] = [config['plm_size'], config['embedding_size']]
        pretrain_data = CustomizedTrainDataLoader(config, pretrain_dataset, None, shuffle=True)
        # model loading and initialization
        model = UniSRec(config, pretrain_data.dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = TunePretrainTrainer(config, model, TRIAL_CNT+1)

        # model pre-training
        trainer.pretrain(pretrain_data, show_progress=True)
        preds, truth = trainer.model.cand_sort_predict(pretrain_val_dataset.plm_embedding, pretrain_val_data, k=5)
        metrics = accuracy_calculator(preds, truth, ACC_KPI)
        kpi = metrics[0]
        logger.info(f"Finish {TRIAL_CNT+1} trial for UniSRec...")
        TRIAL_CNT += 1

        return kpi

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=config['seed']))
    study.optimize(objective, n_trials=args.trials)
    tune_params = list(set(tune_params))
    tune_log_path = f'./tune_log/sample_150/{args.d}/'
    if not os.path.exists(tune_log_path):
        os.makedirs(tune_log_path)
    res_csv = tune_log_path + f'result_{args.d}_UniSRec.csv'
    with open(res_csv, 'w', newline='') as f:
        fieldnames = ['Trial ID'] + tune_params + ['NDCG@5']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for estudio in study.trials:
            w_dict = {}
            w_dict['Trial ID'] = estudio.number+1
            for paras in tune_params:
                w_dict[paras] = estudio.params[paras]
            w_dict['NDCG@5'] = estudio.value
            writer.writerow(w_dict)
        best_dict = {}
        best_dict['Trial ID'] = study.best_trial.number+1
        best_dict['NDCG@5'] = study.best_value
        for paras in tune_params:
            best_dict[paras] = study.best_trial.params[paras]
        writer.writerow(best_dict)
        f.flush()
        f.close()
    logger.info(f"Best trial for UniSRec: {study.best_trial.number+1}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='mg', help='dataset name, options:mg/mb/gb')
    parser.add_argument('-trials', type=int, default=50)
    args, unparsed = parser.parse_known_args()
    pretrain(args)
