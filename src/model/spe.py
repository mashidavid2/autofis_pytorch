import torch
from src.classes.model_info import GMFInfo, SDAEInfo, SPEInfo
from src.model.gmf import GMF
from src.model.sdae import StackedAutoEncoder


class SPE(torch.nn.Module):
    def __init__(self, model_info: SPEInfo, gmf_info: GMFInfo, sdae_info: SDAEInfo):
        super(SPE, self).__init__()
        self.model_info = model_info
        self.gmf_info = gmf_info
        self.sdae_info = sdae_info
        self.gmf = GMF(gmf_info)
        self.sdae = StackedAutoEncoder(self.sdae_info)

        self.embedding_item_gmf = torch.nn.Embedding(
            num_embeddings=self.model_info.num_items, embedding_dim=self.gmf_info.latent_dim)
        self.embedding_item_sdae = torch.nn.Embedding(
            num_embeddings=self.model_info.num_items, embedding_dim=self.gmf_info.latent_dim)

        self.delta_fc = torch.nn.Linear(in_features=self.model_info.num_items, out_features=1)
        self.delta_layer = torch.nn.Sequential(
            self.delta_fc,
            torch.nn.Sigmoid()
        )

        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices1, item_indices2):
        item_embedding_ncf_target = self.embedding_item_ncf(item_indices1)
        item_embedding_sdae_target = self.embedding_item_sdae(item_indices1)
        item_embedding_ncf_others = self.embedding_item_ncf(item_indices2)
        item_embedding_sdae_others = self.embedding_item_sdae(item_indices2)

        delta_target = self.delta_layer(item_embedding_ncf_target).view(-1, 1)
        delta_others = self.delta_layer(item_embedding_ncf_others).view(-1, 1)

        item_embedding_target = \
            item_embedding_ncf_target * (torch.ones(delta_target.shape) - delta_target)\
            + item_embedding_sdae_target * delta_target
        item_embedding_others = \
            item_embedding_ncf_others * (torch.ones(delta_target.shape) - delta_others)\
            + item_embedding_sdae_others * delta_others

        dots = torch.inner(item_embedding_target, item_embedding_others).sum(dim=-1).view(-1, 1)
        output = self.logistic(dots)
        return output

    def load_pretrain_weight(self, base_dir: str):
        self.resume_checkpoint(self.ncf, base_dir, self.ncf_info.model_name)
        self.resume_checkpoint(self.sdae, base_dir, self.sdae_info.model_name)

    def resume_checkpoint(self, model, base_dir, model_name, device_id=0):
        state_dict = torch.load('{}/{}/pth'.format(base_dir, model_name))
        model.load_state_dict(state_dict)


class LinearWeightedAvg(torch.nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedAvg, self).__init__()
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(1)) for _ in range(n_inputs)])

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        pass


# class SPEEngine(Engine):
#     def __init__(self, model_info: SPEInfo, gmf_info: GMFInfo, sdae_info: SDAEInfo, column_info: ColumnInfo, base_dir: str, csv_path: str):
#         super(SPEEngine, self).__init__(model_info, column_info, base_dir)
#         self.model_info, self.gmf_info, self.sdae_info = self._model_info(model_info, gmf_info, sdae_info)
#         self.model = SPE(self.model_info, self.gmf_info, self.sdae_info)
#         self.sdae = StackedAutoEncoder(self.sdae_info)
#         self.gmf = GMF(self.gmf_info)
#         self.crit = torch.nn.BCELoss()
#         self.opt = torch.optim.SGD(self.model.parameters(), lr=model_info.learning_rate)
#
#     def _model_info(self, model_info: SPEInfo, gmf_info: GMFInfo, sdae_info: SDAEInfo) -> Tuple[SPEInfo, GMFInfo, SDAEInfo]:
#         new_model_info = model_info
#
#         new_gmf_info = gmf_info
#
#         new_sdae_info = sdae_info
#         new_sdae_info.input_layer = self.sample_generator.vectorizer.item_vector_size
#         assert new_gmf_info.latent_dim == new_sdae_info.latent_layer
#         return new_model_info, new_gmf_info, new_sdae_info
#
#     def _train(self) -> Evaluation:
#         test_loss = {}
#         for epoch in range(self.model_info.epoch):
#             train_loader = self.sample_generator.instance_a_train_loader(
#                 self.model_info.batch_size)
#             device = torch.device('cpu')
#             self.model.to(device)
#             self._train_an_epoch(train_loader, epoch)
#             test_loss[epoch] = float(self._test_an_epoch(epoch))
#         return Evaluation(loss=test_loss[self.model_info.epoch - 1])
#
#     def _train_an_epoch(self, train_loader, epoch_id):
#         self.model.train()
#         # total_loss = 0
#         for batch_id, batch in enumerate(train_loader):
#             item1_idx, item2_idx, item1, item2, similarity = batch
#             with torch.no_grad():
#                 item1_embedding_sdae = self.sdae.encode(item1)
#                 item2_embedding_sdae = self.sdae.encode(item2)
#                 item1_embedding_gmf = self.gmf.embedding_item(item1_idx)
#                 item2_embedding_gmf = self.gmf.embedding_item(item2_idx)
#
#             loss = self._train_single_batch(item1, item2, similarity)
#             # total_loss += loss
#
#     def _train_single_batch(self, item1, item2, similarity) -> float:
#         # self.opt.zero_grad()
#         # similarity_pred = self.model()
#         pass
#
#     def _test_an_epoch(self, epoch_id) -> float:
#         pass
#
#     def _evaluate(self) -> Evaluation:
#         pass
#
#     def save(self):
#         pass
