import torch
from src.model.gmf import GMF
from src.model.mlp import MLP
from src.model.behavior import BehaviorEngine
from src.classes.model_name import ModelName
from src.classes.model_info import NCFInfo, GMFInfo, MLPInfo
from src.classes.column_info import ColumnInfo


class NCF(torch.nn.Module):
    def __init__(self, model_info: NCFInfo):
        super(NCF, self).__init__()
        self.model_info = model_info
        self.num_users = model_info.num_users
        self.num_items = model_info.num_items
        self.latent_dim_gmf = model_info.latent_dim_gmf
        self.latent_dim_mlp = model_info.latent_dim_mlp

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_gmf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_gmf)
        self.embedding_item_gmf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_gmf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(model_info.layers[:-1], model_info.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=model_info.layers[-1] + self.latent_dim_gmf, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_gmf = self.embedding_user_gmf(user_indices)
        item_embedding_gmf = self.embedding_item_gmf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        gmf_vector = torch.mul(user_embedding_gmf, item_embedding_gmf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, gmf_vector], dim=-1)
        logits = self.affine_output(vector)
        output = self.logistic(logits)
        return output

    def init_weight(self):
        pass

    def load_pretrain_weights(self, base_dir: str):
        """Loading weights from trained MLP model & GMF model"""
        mlp_info = MLPInfo(model_name=ModelName.MLP,
                           latent_dim=self.model_info.latent_dim_mlp,
                           layers=self.model_info.layers)
        mlp_info.set_users_items(self.num_users, self.num_items)

        mlp_model = MLP(mlp_info)
        # if config['use_cuda'] is True:
        #     mlp_model.cuda()
        self.resume_checkpoint(mlp_model, base_dir, mlp_info.model_name)

        self.embedding_user_mlp.weight.data = mlp_model.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp_model.embedding_item.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp_model.fc_layers[idx].weight.data

        gmf_info = GMFInfo(model_name=ModelName.GMF,
                           latent_dim=self.model_info.latent_dim_gmf)
        gmf_info.set_users_items(self.num_users, self.num_items)

        gmf_model = GMF(gmf_info)
        # if config['use_cuda'] is True:
        #     gmf_model.cuda()
        self.resume_checkpoint(gmf_model, base_dir, gmf_info.model_name)
        self.embedding_user_gmf.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item_gmf.weight.data = gmf_model.embedding_item.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat([mlp_model.affine_output.weight.data, gmf_model.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp_model.affine_output.bias.data + gmf_model.affine_output.bias.data)

    def resume_checkpoint(self, model, base_dir, model_name, device_id=0):
        state_dict = torch.load('{}/{}.pth'.format(base_dir, model_name))
        # state_dict = torch.load(base_dir,
        #                         map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
        model.load_state_dict(state_dict)


class NCFEngine(BehaviorEngine):
    def __init__(self, model_info: NCFInfo, column_info: ColumnInfo, base_dir: str, csv_path: str):
        super(NCFEngine, self).__init__(model_info, column_info, base_dir, csv_path)
        self.model.load_pretrain_weights(base_dir)

    def _model(self, model_info: NCFInfo):
        return NCF(model_info)

    def _opt(self, model_info: NCFInfo):
        return torch.optim.SGD(self.model.parameters(),
                               lr=model_info.learning_rate,
                               weight_decay=model_info.regularization)
