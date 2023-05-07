# class BehaviorEngine(Engine):
#     def __init__(self, model_info: ModelInfo, column_info: ColumnInfo, base_dir: str, csv_path: str):
#         super(BehaviorEngine, self).__init__(model_info, column_info, base_dir)
#         self.model_info = self._model_info(model_info)
#         self.model = self._model(model_info)
#         self.opt = self._opt(model_info)
#         self.crit = torch.nn.BCELoss()
#         # if config['use_cuda'] is True:
#         #     use_cuda(True, config['device_id'])
#         #     self.model.cuda()
#
#     def _model_info(self, model_info: ModelInfo) -> ModelInfo:
#         new_model_info = model_info
#         # new_model_info.set_users_items(
#         #     len(self.sample_generator.user_pool),
#         #     len(self.sample_generator.item_pool)
#         # )
#         return new_model_info
#
#     @abc.abstractmethod
#     def _model(self, model_info: ModelInfo) -> torch.nn.Module:
#         raise NotImplementedError('model must define in each behavior model')
#
#     @abc.abstractmethod
#     def _opt(self, model_info: ModelInfo):
#         raise NotImplementedError('optimizer must define in each behavior model')
#
#     def _train(self) -> Evaluation:
#         test_loss = {}
#         for epoch in range(self.model_info.epoch):
#             train_loader = self.sample_generator.instance_a_train_loader(
#                 self.model_info.num_negative, self.model_info.batch_size)
#             device = torch.device('cpu')
#             self.model.to(device)
#             self._train_an_epoch(train_loader, epoch_id=epoch)
#             test_loss[epoch] = float(self._test_an_epoch(epoch))
#         return Evaluation(loss=test_loss[self.model_info.epoch - 1])
#
#     def _train_an_epoch(self, train_loader, epoch_id):
#         self.model.train()
#         total_loss = 0
#         for batch_id, batch in enumerate(train_loader):
#             assert isinstance(batch[0], torch.LongTensor)
#             user, item, rating = batch[0], batch[1], batch[2]
#             rating = rating.float()
#             loss = self._train_single_batch(user, item, rating)
#             # print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
#             total_loss += loss
#         # self._writer.add_scalar('model/loss', total_loss, epoch_id)
#         # self._writer.add_scalar('model/loss', total_loss/(batch_id+1), epoch_id)
#
#     def _train_single_batch(self, users, items, ratings) -> float:
#         # if self.config['use_cuda'] is True:
#         #     users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
#         self.opt.zero_grad()
#         ratings_pred = self.model(users, items)
#         loss = self.crit(ratings_pred.view(-1), ratings)
#         loss.backward()
#         self.opt.step()
#         loss = loss.item()
#         return loss
#
#     def _test_an_epoch(self, epoch_id) -> float:
#         test_loader = self.sample_generator.instance_a_test_loader()
#         with torch.no_grad():
#             for batch_id, batch in enumerate(test_loader):
#                 assert isinstance(batch[0], torch.LongTensor)
#                 user, item, rating = batch[0], batch[1], batch[2]
#                 ratings_pred = self.model(user, item)
#                 test_loss = self.crit(ratings_pred.view(-1), rating)
#                 print('[Test Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, test_loss))
#         return test_loss
#
#     def _evaluate(self) -> Evaluation:
#         evaluation_loader: DataLoader = self.sample_generator.instance_a_test_loader()
#         negative_loader: DataLoader = self.sample_generator.instance_a_negative_loader()
#         user_col = self.column_info.get_user_name()
#         item_col = self.column_info.get_item_name()
#         cols = [user_col, item_col, 'score']
#
#         test_pred_df = pd.DataFrame(columns=cols)
#         neg_pred_df = pd.DataFrame(columns=cols)
#         with torch.no_grad():
#             for batch_id, batch in enumerate(evaluation_loader):
#                 assert isinstance(batch[0], torch.LongTensor)
#                 user, item, rating = batch[0], batch[1], batch[2]
#                 test_pred = self.model(user, item)
#                 test_pred_df = test_pred_df.append(
#                     pd.DataFrame(
#                         zip(user.tolist(), item.tolist(), test_pred.view(-1).tolist()),
#                         columns=cols))
#
#             for batch_id, batch in enumerate(negative_loader):
#                 assert isinstance(batch[0], torch.LongTensor)
#                 user, item, rating = batch[0], batch[1], batch[2]
#                 negative_pred = self.model(user, item)
#                 neg_pred_df = neg_pred_df.append(
#                     pd.DataFrame(
#                         zip(user.tolist(), item.tolist(), negative_pred.view(-1).tolist()),
#                         columns=cols))
#         test_pred_df.reset_index(inplace=True, drop=True)
#         neg_pred_df.reset_index(inplace=True, drop=True)
#         return self._metric.evaluate(test_pred_df, neg_pred_df)
#
#     def save(self):
#         torch.save(self.model.state_dict(), '{}/{}.pth'.format(self.base_dir, self.model_info.model_name))
