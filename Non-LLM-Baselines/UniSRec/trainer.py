import os
from time import time
from recbole.trainer import Trainer, PretrainTrainer
from recbole.utils import set_color


class TunePretrainTrainer(PretrainTrainer):
    def __init__(self, config, model, trial):
        super().__init__(config, model)
        self.trial = trial
    def pretrain(self, train_data, verbose=True, show_progress=False):
        last_loss = 0.0
        for epoch_idx in range(self.start_epoch, self.pretrain_epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple)
                else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)

            if (epoch_idx + 1) % self.save_step == 0:
                saved_model_file = os.path.join(
                    self.checkpoint_dir,
                    "{}-{}-{}-{}.pth".format(
                        self.config["model"], self.config["dataset"], str(epoch_idx + 1), self.trial
                    ),
                )
                self.save_pretrained_model(epoch_idx, saved_model_file)

                update_output = (
                    set_color("Saving current", "blue") + ": %s" % saved_model_file
                )
                if verbose:
                    self.logger.info(update_output)
            delta_loss = train_loss - last_loss
            if abs(delta_loss) < 1e-5:
                print(f'early stop at epoch {epoch_idx}')
                break
            last_loss = delta_loss
            if (epoch_idx + 1) > 1:
                del_model_file = os.path.join(
                    self.checkpoint_dir,
                    "{}-{}-{}-{}.pth".format(
                        self.config["model"], self.config["dataset"], str(epoch_idx), self.trial
                    ),
                )
                os.remove(del_model_file)

        return self.best_valid_score, self.best_valid_result


class FinetuneTrainer(Trainer):
    def __init__(self, config, model):
        super().__init__(config, model)
    
    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        last_loss = 0.0
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )
            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )
            delta_loss = train_loss - last_loss
            if abs(delta_loss) < 1e-5:
                print(f'early stop at epoch {epoch_idx}')
                break
            last_loss = delta_loss

        return self.best_valid_score, self.best_valid_result