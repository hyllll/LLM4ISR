import torch
from recbole.data.dataloader.general_dataloader import TrainDataLoader, FullSortEvalDataLoader
from data.transform import construct_transform


class CustomizedTrainDataLoader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.transform = construct_transform(config)

    def _next_batch_data(self):
        cur_data = super()._next_batch_data()
        transformed_data = self.transform(self, cur_data)
        return transformed_data

class CustomizedFullSortEvalDataLoader(FullSortEvalDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)
        self.transform = construct_transform(config)

    def _next_batch_data(self):
        cur_data = super()._next_batch_data()
        transformed_data = self.transform(self, cur_data[0])
        return (transformed_data, *cur_data[1:])

def evaluate_collate_fn(batch, max_length):
    seqs, seqs_length, targets, cands = [torch.empty(max_length)], [], [], []
    for seq, seq_length, target, cand in batch:
        seqs += [torch.tensor(seq)]
        seqs_length += [torch.tensor(seq_length)]
        targets += [torch.tensor(target)]
        cands += [torch.tensor(cand)]
    
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.)
    seqs = seqs[1:]
    seqs = seqs.type(torch.int64)
    seqs_length = torch.stack(seqs_length)
    targets = torch.stack(targets)
    cands = torch.stack(cands)

    return seqs, seqs_length, targets, cands