import torch
from src.classes.model_info import GMFInfo


# from src.behavior import BehaviorEngine


class GMF(torch.nn.Module):
    def __init__(self, model_info: GMFInfo):
        super(GMF, self).__init__()
        self.num_users = model_info.num_users
        self.num_items = model_info.num_items
        self.latent_dim = model_info.latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


# class GMFEngine(BehaviorEngine):
#     """Engine for training & evaluating GMF model"""
#     def __init__(self, model_info: GMFInfo, column_info: ColumnInfo, base_dir: str, csv_path: str):
#         super(GMFEngine, self).__init__(model_info, column_info, base_dir, csv_path)
#
#     def _model(self, model_info: GMFInfo):
#         return GMF(self.model_info)
#
#     def _opt(self, model_info: GMFInfo):
#         return torch.optim.Adam(self.model.parameters(),
#                                 lr=model_info.learning_rate,
#                                 weight_decay=model_info.regularization)
