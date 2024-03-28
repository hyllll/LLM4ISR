import os
import torch
from recbole.config import Config as RecBoleConfig

class Config(RecBoleConfig):
    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu and ('ddp' in self.final_config_dict and not self.final_config_dict['ddp']):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


HyperParameter_setting = {
    'UniSRec': {
        'categorical': {
            'hidden_size': [32, 64, 128],
            'learning_rate': [0.0001, 0.001, 0.01],
            'train_batch_size': [64, 128, 256],
        }
    },
}

Best_setting = {
    'bundle':{
        'train_batch_size': 128,
        'embedding_size': 128,
        'learning_rate': 0.001,
    },
    'games':{
        'train_batch_size': 64,
        'embedding_size': 128,
        'learning_rate': 0.001,
    },
    'ml':{
        'train_batch_size': 64,
        'embedding_size': 128,
        'learning_rate': 0.0001,
    }
}