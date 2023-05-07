import torch
from src.classes.model_info import MLPInfo


# from src.behavior import BehaviorEngine


class MLP(torch.nn.Module):
    def __init__(self, model_info: MLPInfo):
        super(MLP, self).__init__()
        self.num_users = model_info.num_users
        self.num_items = model_info.num_items
        self.latent_dim = model_info.latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(model_info.layers[:-1], model_info.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=model_info.layers[-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        logits = self.affine_output(vector)
        output = self.logistic(logits)
        return output

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        # config = self.config
        # gmf_model = GMF(config)
        # if config['use_cuda'] is True:
        #     gmf_model.cuda()
        # resume_checkpoint(gmf_model, base_dir=config['pretrain_mf'], device_id=config['device_id'])
        # self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
        # self.embedding_item.weight.data = gmf_model.embedding_item.weight.data
        pass


# class MLPEngine(BehaviorEngine):
#     def __init__(self, model_info: MLPInfo, column_info: ColumnInfo, base_dir: str, csv_path: str):
#         super(MLPEngine, self).__init__(model_info, column_info, base_dir, csv_path)
#
#     def _model(self, model_info: MLPInfo) -> torch.nn.Module:
#         return MLP(self.model_info)
#
#     def _opt(self, model_info: MLPInfo):
#         return torch.optim.Adam(self.model.parameters(),
#                                 lr=model_info.learning_rate,
#                                 weight_decay=model_info.regularization)

