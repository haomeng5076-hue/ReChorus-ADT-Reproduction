# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np
from models.BaseModel import GeneralModel

class ADT(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['denoise_drop_rate', 'warm_up_epoch']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--denoise_drop_rate', type=float, default=0.1,
                            help='The ratio of noise to drop (beta in paper).')
        parser.add_argument('--warm_up_epoch', type=int, default=5,
                            help='Number of epochs to train normally before denoising.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.drop_rate = args.denoise_drop_rate
        self.warm_up_epoch = args.warm_up_epoch
        self.user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.item_emb = nn.Embedding(self.item_num, self.emb_size)
        self.apply(self.init_weights)

    def forward(self, feed_dict):
        u_ids = feed_dict['user_id']
        i_ids = feed_dict['item_id']
        u_vectors = self.user_emb(u_ids)
        i_vectors = self.item_emb(i_ids)

        prediction = (u_vectors[:, None, :] * i_vectors).sum(dim=-1)

        out_dict = feed_dict.copy()
        out_dict['prediction'] = prediction.view(feed_dict['batch_size'], -1)
        
        return out_dict

def loss(self, out_dict):
        # 1. 获取输入
        u_ids = out_dict['user_id']
        pos_ids = out_dict['item_id']
        
        if pos_ids.dim() > 1:
            pos_ids = pos_ids[:, 0]

        # 2. 生成负样本
        batch_size = u_ids.shape[0]
        neg_ids = torch.randint(0, self.item_num, (batch_size,), device=u_ids.device)

        # 3. 计算分数 (使用 view 确保形状安全)
        u_vectors = self.user_emb(u_ids)       # [batch, emb]
        pos_vectors = self.item_emb(pos_ids)   # [batch, emb]
        neg_vectors = self.item_emb(neg_ids)   # [batch, emb]

        pos_scores = (u_vectors * pos_vectors).sum(dim=-1)
        neg_scores = (u_vectors * neg_vectors).sum(dim=-1)

        # 4. 计算 BCE Loss
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        per_sample_loss = criterion(all_scores, all_labels)

        # 5. 去噪逻辑
        num_samples = per_sample_loss.shape[0]
        num_keep = int(num_samples * (1 - self.drop_rate))
        
        if num_keep > 0:
            kept_loss, _ = torch.topk(per_sample_loss, k=num_keep, largest=False)
            final_loss = kept_loss.mean()
        else:
            final_loss = per_sample_loss.mean()

        return final_loss
