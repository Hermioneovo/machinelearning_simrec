import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
from models.general.LightGCN import LightGCN
import os


class SimRec(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    # 移除teacher_path，只保留其他参数
    extra_log_args = ['emb_size', 'lambda_kd', 'temp_soft', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        # 首先调用父类的parse_model_args来添加通用参数
        parser = GeneralModel.parse_model_args(parser)
        # 然后添加SimRec特有的参数
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='Number of LightGCN layers (for teacher model).')
        parser.add_argument('--teacher_path', type=str, default='',
                            help='Path of teacher model.')
        parser.add_argument('--lambda_kd', type=float, default=0.3,
                            help='Weight of knowledge distillation loss.')
        parser.add_argument('--temp_soft', type=float, default=1.0,
                            help='Temperature for soft label.')
        return parser

    def __init__(self, args, corpus):
        super().__init__(args, corpus)

        self.latdim = args.emb_size
        self.user_num = self.user_num
        self.item_num = self.item_num

        # student
        self.user_emb = nn.Embedding(self.user_num, self.latdim)
        self.item_emb = nn.Embedding(self.item_num, self.latdim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        # 为教师模型准备参数
        teacher_args = type(args)(**vars(args))  # 创建args的副本

        # teacher（固定，不训练）
        self.teacher = LightGCN(teacher_args, corpus)
        if args.teacher_path and os.path.exists(args.teacher_path):
            # 从teacher_path中提取模型名称用于日志
            teacher_filename = os.path.basename(args.teacher_path)
            self.teacher_name = teacher_filename.replace('.pt', '').replace('LightGCN__', '')

            try:
                self.teacher.load_state_dict(
                    torch.load(args.teacher_path, map_location='cpu')
                )
            except Exception as e:
                print(f"Warning: Failed to load teacher model: {e}")
                print("Teacher model will be used without pre-training.")
        else:
            self.teacher_name = 'no_teacher'
            print(f"Warning: Teacher model path not found: {args.teacher_path}")
            print("Teacher model will be used without pre-training.")

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.temp_soft = args.temp_soft
        self.lambda_kd = args.lambda_kd

    # ---------- forward ----------
    def forward(self, feed_dict):
        user = feed_dict['user_id']
        item = feed_dict['item_id']

        # 确保数据类型正确
        user = user.long() if user is not None else None
        item = item.long() if item is not None else None

        u = self.user_emb(user)  # [batch_size, emb_size]

        # 获取item的维度信息
        item_shape = item.shape
        if len(item_shape) == 1:
            # 训练时：item是单个物品ID [batch_size]
            i = self.item_emb(item)  # [batch_size, emb_size]
            prediction = (u * i).sum(dim=-1)  # [batch_size]
        elif len(item_shape) == 2:
            # 评估时：item是多个物品ID [batch_size, num_items]
            # 重塑item为 [batch_size * num_items]
            item_flat = item.view(-1)
            i_flat = self.item_emb(item_flat)  # [batch_size * num_items, emb_size]

            # 扩展user嵌入
            u_expanded = u.unsqueeze(1).repeat(1, item_shape[1], 1)  # [batch_size, num_items, emb_size]
            u_flat = u_expanded.view(-1, u.shape[-1])  # [batch_size * num_items, emb_size]

            # 计算分数并重塑回原始形状
            scores_flat = (u_flat * i_flat).sum(dim=-1)  # [batch_size * num_items]
            prediction = scores_flat.view(item_shape)  # [batch_size, num_items]
        else:
            raise ValueError(f"Unexpected item shape: {item_shape}")

        out_dict = {'prediction': prediction}

        if feed_dict['phase'] == 'train':
            out_dict['loss'] = self.calculate_loss(feed_dict)

        return out_dict

    # ---------- loss ----------
    def calculate_loss(self, feed_dict):
        user = feed_dict['user_id']
        pos_item = feed_dict['item_id']

        # 确保是训练模式（单个正样本）
        if len(pos_item.shape) != 1:
            # 如果是评估模式，返回0损失
            return torch.tensor(0.0, device=user.device)

        # student scores for positive items
        pos_score = (self.user_emb(user) * self.item_emb(pos_item)).sum(dim=-1)  # [batch_size]

        # 生成负样本
        batch_size = user.shape[0]
        neg_items = torch.randint(0, self.item_num, (batch_size,), device=user.device)
        neg_score = (self.user_emb(user) * self.item_emb(neg_items)).sum(dim=-1)  # [batch_size]

        # BPR loss
        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

        # teacher scores
        with torch.no_grad():
            # 准备教师模型的输入
            all_items = torch.stack([pos_item, neg_items], dim=1)  # [batch_size, 2]
            teacher_feed = {
                'user_id': user,
                'item_id': all_items,
                'batch_size': batch_size,
                'phase': 'train'
            }
            teacher_out = self.teacher(teacher_feed)
            teacher_scores = teacher_out['prediction']

            # 确保教师模型的输出形状正确
            if len(teacher_scores.shape) == 1:
                # 如果教师模型返回的是展平的分数
                teacher_scores = teacher_scores.view(batch_size, -1)

            t_pos = teacher_scores[:, 0]
            t_neg = teacher_scores[:, 1]

        # KD loss
        student_scores = torch.stack([pos_score, neg_score], dim=1)  # [batch_size, 2]
        teacher_scores = torch.stack([t_pos, t_neg], dim=1)  # [batch_size, 2]

        kd_loss = F.kl_div(
            F.log_softmax(student_scores / self.temp_soft, dim=1),
            F.softmax(teacher_scores / self.temp_soft, dim=1),
            reduction='batchmean'
        )

        return bpr_loss + self.lambda_kd * kd_loss