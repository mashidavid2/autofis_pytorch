import torch
from src.classes.model_info import SDAEInfo


class StackedAutoEncoder(torch.nn.Module):
    def __init__(self, model_info: SDAEInfo):
        super(StackedAutoEncoder, self).__init__()

        self.autoencoders = torch.nn.ModuleList()

        # TODO Engine에서 dataset -> in_feature 값 추출
        in_features = model_info.input_layer
        layers = [in_features] + model_info.hidden_layers + [model_info.latent_layer]
        for _, (in_size, out_size) in enumerate(zip(layers, layers[1:])):
            self.autoencoders.append(AutoEncoder(in_size, out_size, model_info.dropout))

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed

    def encode(self, x):
        for i, ae in enumerate(self.autoencoders):
            x = ae.encode(x)

            if i != len(self.autoencoders) - 1:
                x = ae.dropout(x)
        return x

    def decode(self, x):
        for i, ae in enumerate(reversed(self.autoencoders)):
            if i != 0:
                x = ae.activation(x)

            x = ae.decode(x)
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(self, in_features, latent_size, dropout, tie_weights=True):
        super(AutoEncoder, self).__init__()

        encode = torch.nn.Linear(in_features, latent_size)
        decode = torch.nn.Linear(latent_size, in_features)

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.Sigmoid()

        if tie_weights:
            decode.weight.data = encode.weight.data.t()

        self.encode = torch.nn.Sequential(
            encode,
            self.activation
        )

        self.decode = torch.nn.Sequential(
            self.dropout,
            decode
        )

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return latent, reconstructed


# class SDAEEngine(Engine):
#     def __init__(self, model_info: SDAEInfo, column_info: ColumnInfo, base_dir: str, csv_path: str):
#         super(SDAEEngine, self).__init__(model_info, column_info, base_dir)
#         self.model_info = model_info
#         self.model = StackedAutoEncoder(model_info)
#         self.column_info = column_info
#         self.opt = torch.optim.AdamW(
#             self.model.parameters(),
#             lr=model_info.learning_rate,
#             weight_decay=model_info.regularization
#         )
#         self.crit = torch.nn.MSELoss()
#
#     def _model_info(self, model_info: SDAEInfo):
#         new_model_info = model_info
#         new_model_info.input_layer = self.sample_generator.vectorizer.vector_size
#
#     def _train(self) -> Evaluation:
#         device = torch.device('cpu')
#         self.model.to(device)
#
#         cur_dataset = self.sample_generator.get_train_tensors()
#
#         for i, autoencoder in enumerate(self.model.autoencoders):
#             print('Layer-wise Train {}th autoencoder'.format(i+1))
#             self._train_isolated_autoencoder(autoencoder, cur_dataset)
#
#             with torch.no_grad():
#                 autoencoder.eval()
#                 cur_dataset = autoencoder.encode(cur_dataset)
#                 autoencoder.train()
#
#         print('Fitting stacked autoencoder')
#         cur_dataset = self.sample_generator.get_train_tensors()
#         test_loss = self._train_isolated_autoencoder(self.model, cur_dataset)
#
#         return Evaluation(loss=test_loss)
#
#     def _train_isolated_autoencoder(self, autoencoder, content) -> float:
#         test_loss = {}
#         for epoch in range(self.model_info.epoch):
#             dataset = TransformDataset(
#                 content,
#                 lambda x: (self._bernoulli_corrupt(x, self.model_info.corruption), x)
#             )
#             self._train_an_epoch(autoencoder, dataset, epoch)
#             test_loss[epoch] = float(self._test_an_epoch(autoencoder, dataset, epoch))
#         return test_loss[self.model_info.epoch - 1]
#
#     def _train_an_epoch(self, autoencoder, dataset, epoch_id):
#         total_loss = 0
#         train_loader = DataLoader(dataset, self.model_info.batch_size)
#         for batch_id, batch in enumerate(train_loader):
#             x, y = batch[0], batch[1]
#             loss = self._train_single_batch(autoencoder, x, y)
#             total_loss += loss
#         print('[Training Epoch {}] Loss {}'.format(epoch_id, total_loss / (batch_id + 1)))
#
#     def _train_single_batch(self, autoencoder, x, y):
#         self.opt.zero_grad()
#         _, y_pred = autoencoder(x)
#         loss = self.crit(y_pred, y)
#         loss.backward()
#         self.opt.step()
#         loss = loss.item()
#         return loss
#
#     def _test_an_epoch(self, autoencoder, dataset, epoch_id) -> float:
#         test_loss = 0
#         test_loader = DataLoader(dataset, self.model_info.batch_size)
#         with torch.no_grad():
#             for batch_id, batch in enumerate(test_loader):
#                 x, _ = batch[0], batch[1]
#                 _, pred = autoencoder(x)
#                 loss = self.crit(pred, x)
#                 test_loss += loss
#             print('[Test Epoch {}], Loss {}'.format(epoch_id, test_loss / (batch_id + 1)))
#         return test_loss / (batch_id + 1)
#
#     def _bernoulli_corrupt(self, x, p):
#         mask = torch.rand_like(x) > p
#         return x * mask
#
#     # def _loss_fn(self, pred, target):
#     #     latent_pred, recon_pred = pred
#     #     latent_target, recon_target = target
#     #     return self.crit(recon_pred, recon_target) + self.crit(latent_pred, latent_target)
#
#     def _evaluate(self) -> Evaluation:
#         return Evaluation()
#
#     def save(self):
#         torch.save(self.model.state_dict(), '{}/{}.pth'.format(self.base_dir, self.model_info.model_name))
