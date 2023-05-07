import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time, os
import datetime
import pandas as pd
from typing import Optional

from engine import Engine
from model import AutoDeepFM
from classes import AutoFisInfo, ColumnInfo, Evaluation
from utils import EarlyStopping, Timer, GRDA, get_loss, get_l2_loss

class AutoFisEngine(Engine):
    def __init__(self, model_info: AutoFisInfo, column_info: ColumnInfo, base_dir: str, save_dir: str):
        super(AutoFisEngine, self).__init__(model_info, column_info, base_dir, save_dir)
        self.model: Optional[AutoDeepFM] = None
        self.early_stopping = EarlyStopping(patience=10, mode='max', save_callback=self._save, verbose=True)

        self.batch_size = model_info.batch_size
        self._learning_rate = model_info.learning_rate
        self.decay_rate = model_info.decay_rate
        self._learning_rate2 = model_info.learning_rate2
        self.decay_rate2 = model_info.decay_rate2
        self.epsilon = 1e-8
        self.grda_c = model_info.grda_c
        self.grda_mu = model_info.grda_mu
        self.n_epoch = model_info.epoch
        self.save_dir = save_dir
        self.saver = None

    def _init_model(self, retrain_stage=0, comb_mask=None, comb_mask_third=None):
        torch.cuda.empty_cache()
        torch.manual_seed(2023)
        torch.cuda.manual_seed(2023)
        np.random.seed(2023)

        # general parameter
        depth = 5
        width = 700
        ls = [width] * depth
        ls.append(1)
        la = ['relu'] * depth
        la.append(None)
        lk = [1.0] * depth
        lk.append(1.)
        third_prune = True
        if self.feature_info.feature_nums < 3:
            third_prune = False

        self.retrain_stage = retrain_stage
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoDeepFM(
            num_inputs=self.feature_info.feature_nums, input_dim=self.feature_info.feature_total_size,
            l2_v=0.0, layer_sizes=ls, layer_acts=la, layer_keeps=lk, layer_l2=[0, 0],
            embed_size=self.model_info.latent_dim, retrain_stage=retrain_stage,
            comb_mask=comb_mask, comb_mask_third=comb_mask_third, third_prune=third_prune, batch_size =self.batch_size, device=self.device
        )
        self.model.to(self.device)

        if self.retrain_stage:
            self.optimizer1 = optim.Adam(self.model.parameters(), lr=self._learning_rate, eps=self.epsilon)
            self.scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer1, factor=self.decay_rate, patience=1)
        else:
            if not third_prune:
                deep_models = list(self.model.xw_embed.parameters())+list(self.model.xv_embed.parameters())+list(self.model.bin_mlp.parameters())+list(self.model.linear.parameters())
                matrix_models = list(self.model.level_2_matrix.parameters())
            else:
                deep_models = list(self.model.xw_embed.parameters())+list(self.model.xv_embed.parameters())+list(self.model.xps_embed.parameters())+list(self.model.bin_mlp.parameters())+list(self.model.linear.parameters())
                matrix_models = list(self.model.level_2_matrix.parameters())+list(self.model.level_3_matrix.parameters())
            self.optimizer1 = optim.Adam(deep_models, lr=self._learning_rate, eps=self.epsilon)
            self.optimizer2 = GRDA(matrix_models, lr=self._learning_rate2, c=self.grda_c, mu=self.grda_mu)
            self.scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer1, factor=self.decay_rate, patience=2000)
            self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer2, factor=self.decay_rate2, patience=2000)

        self.loss_func = get_loss('weight')
        self.epoch = 0
        self.global_step = 0
        _,_ = self.model.analyse_structure(print_full_weight=False)

    def _train(self, dataset_generator) -> Evaluation:
        self.timer = Timer(
            train_size=self.feature_info.train_size,
            batch_size=self.batch_size,
            epoch_size=self.n_epoch * 2 # search & retrain stage
        )
        self.timer.start()
        _, comb_mask, comb_mask_third = self._run_stage(dataset_generator, retrain_stage=0)
        evaluation, _, _ = self._run_stage(dataset_generator, retrain_stage=1, comb_mask=comb_mask, comb_mask_third=comb_mask_third)
        return evaluation

    def _run_stage(self, dataset_generator, retrain_stage=0, comb_mask=None, comb_mask_third=None):
        self._init_model(retrain_stage, comb_mask, comb_mask_third)

        evaluation = self._fit(retrain_stage, dataset_generator)

        if retrain_stage == 0:
            store_info = self.early_stopping.get_store_info()
            comb_mask, comb_mask_third = store_info['comb_mask'], store_info['comb_mask_third']
            self.early_stopping.init()
        return evaluation, comb_mask, comb_mask_third
    
    ### look again at the store info
    def _fit(self, retrain_stage, dataset_generator):
        train_data_param = {'gen_type': 'train', 'batch_size': self.model_info.batch_size}
        train_gen = dataset_generator.batch_generator(train_data_param)
        test_data_param = {'gen_type': 'test', 'batch_size': self.model_info.batch_size, 'shuffle': False}
        test_gen = dataset_generator.batch_generator(test_data_param)

        num_batches_per_epoch = int(torch.ceil(torch.tensor(self.feature_info.train_size / self.batch_size)))

        store_info = None
        epoch_loss = 0
        unit_batch_loss = 0
        unit_batch = 100

        for epoch_id in range(self.n_epoch):
            self.model.train()
            for batch_id, batch_data in enumerate(train_gen):
                X, y = batch_data
                X = torch.from_numpy(X).float().to(self.device)
                y = torch.from_numpy(y).float().to(self.device)

                batch_loss, _ = self._batch_train(X, y)
                epoch_loss += batch_loss
                unit_batch_loss += batch_loss

                if batch_id != 0 and ((batch_id + 1) % unit_batch == 0 or batch_id + 1 == num_batches_per_epoch):
                    avg_unit_batch_loss = unit_batch_loss / (batch_id % unit_batch + 1)
                    elapsed, eta = self.timer(
                        stage='train',
                        early_stop=self.early_stopping.early_stop,
                    )

                    print(
                        f'[Unit Batch Train statistics] stage: {retrain_stage + 1}/2, epoch: {epoch_id + 1}/{self.n_epoch}, '
                        f'batch: {batch_id + 1}/{num_batches_per_epoch}, training_loss: {avg_unit_batch_loss}, '
                        f'elapsed: {str(datetime.timedelta(seconds=elapsed))}, ETA: {str(datetime.timedelta(seconds=eta))}'
                    )

                    unit_batch_loss = 0
                self.scheduler1.step(metrics=batch_loss)
                if not retrain_stage:
                    self.scheduler2.step(metrics=batch_loss)
            avg_epoch_loss = epoch_loss / num_batches_per_epoch
            elapsed, eta = self.timer(stage='info')
            print(
                f'[Train statistics] stage: {retrain_stage + 1}/2, epoch: {epoch_id + 1}/{self.n_epoch}, '
                f'training_loss: {avg_epoch_loss}, '
                f'elapsed: {str(datetime.timedelta(seconds=elapsed))}, ETA: {str(datetime.timedelta(seconds=eta))}\n'
            )
            epoch_loss = 0

            store_info = self._epoch_callback(retrain_stage, epoch_id, test_gen)

            if self.early_stopping.early_stop: #if early stop is True then save model
                self._save()
                return store_info['evaluation']
        return store_info['evaluation']
    
    def _batch_train(self, X, y):
        self.optimizer1.zero_grad()
        if not self.retrain_stage:
            self.optimizer2.zero_grad()

        outputs = self.model(X)
        loss = self.loss_func(outputs, y)
        if self.model.third_prune:
            l2_loss = get_l2_loss([self.model.l2_w,self.model.l2_v,self.model.l2_ps],
                                  [self.model.xw,self.model.xv,self.model.xps])
        else:
            l2_loss = get_l2_loss([self.model.l2_w,self.model.l2_v],
                                  [self.model.xw,self.model.xv])
        if l2_loss is not None:
            loss += l2_loss
        loss.backward()
        
        if self.retrain_stage:
            self.optimizer1.step()
        else:
            self.optimizer1.step()
            self.optimizer2.step()

        # return loss.item(), l2_loss.item(), outputs.cpu().detach().numpy()
        return loss.item(), outputs.cpu().detach().numpy()

    def _evaluate(self, test_gen):
        test_preds = []
        for batch_data in test_gen:
            X, _ = batch_data
            batch_preds = self._batch_evaluate(X)
            test_preds.append(batch_preds)
        test_preds = np.concatenate(test_preds)
        test_preds = np.float64(test_preds)
        test_preds = np.clip(test_preds, 1e-8, 1 - 1e-8)
        return self._metric.evaluate(test_preds)

    def _batch_evaluate(self, X):
        self.model.eval()
        # X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        return outputs.cpu().detach().numpy()

    def _save(self):
        if self.retrain_stage:
            # torch.save(self.model.state_dict(), f'{self.base_dir}/{self.model_info.model_name}.{self.save_postfix}')
            torch.save(self.model.state_dict(), f'{self.save_dir}/{self.model_info.model_name}.{self.save_postfix}')
            
    def _load(self):
        checkpoint_info = torch.load(os.path.join(self.base_dir, "model.{}".format(self.save_postfix)))
        self.model.load_state_dict(checkpoint_info['model_state_dict'])

    #need to check
    def _inference(self, feature: pd.DataFrame) -> np.ndarray:
        inputs, training, outputs = self._get_inference_tensor()

        inputs = torch.tensor(feature.values, dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            scores = self.model(inputs).cpu().numpy()

        return scores
    
    def _epoch_callback(self, retrain_stage, epoch_id, test_gen):
        store_info = {}

        evaluation = self._evaluate(test_gen)

        print('[Model Architecture statistics]')
        comb_mask, comb_mask_third = self.model.analyse_structure(print_full_weight=True)

        store_info['comb_mask'] = comb_mask
        store_info['comb_mask_third'] = comb_mask_third
        store_info['evaluation'] = evaluation

        self.early_stopping(evaluation.get_metric_from_evaluation_type(self.model_info.evaluation_type), store_info)
        if retrain_stage == 0:
            elapsed, eta = self.timer(stage='test', early_stop=self.early_stopping.early_stop, total_epoch=self.n_epoch)
        else:
            elapsed, eta = self.timer(stage='test', early_stop=self.early_stopping.early_stop)
        print(
            f'[Test statistics] stage: {retrain_stage + 1}/2, epoch: {epoch_id + 1}/{self.n_epoch}, '
            f'evaluation: {str(evaluation)}'
            f'elapsed: {str(datetime.timedelta(seconds=elapsed))}, ETA: {str(datetime.timedelta(seconds=eta))}\n'
        )
        return store_info