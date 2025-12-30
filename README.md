# machinelearning_simrec
## 对比实验
### 源码

在main分支

### 运行指令

运行位置：...\ReChorus-master（即data/docs/src文件夹所在处）

python src/main.py --model_name NeuMF --dataset MovieLens_1M --emb_size 32 --lr 0.001 --l2 0 --layers [64] --epoch 50 --early_stop 10 --batch_size 2048 --eval_batch_size 2048 --num_neg 99 --gpu -1 --num_workers 0

python src/main.py --model_name LightGCN --dataset MovieLens_1M --emb_size 32 --n_layers 2 --lr 0.001 --l2 0 --epoch 50 --early_stop 10 --batch_size 2048 --eval_batch_size 2048 --num_neg 99 --gpu -1 --num_workers 0

python src/main.py --model_name SimRec --dataset MovieLens_1M --emb_size 32 --n_layers 2 --lr 0.001 --l2 0 --dropout 0 --teacher_path ""E:\HOMEWORK\大三上\机器学习\理论课大作业\ReChorus-master\model\LightGCN\LightGCN__MovieLens_1M__0__lr=0.001__l2=0.0__emb_size=32__n_la_size=2048.pt"" --lambda_kd 0.6 --temp_soft 2.0 --epoch 50 --early_stop 10 --batch_size 2048 --eval_batch_size 2048 --num_neg 99 --gpu -1 --num_workers 0

python src/main.py --model_name NeuMF --dataset Grocery_and_Gourmet_Food --emb_size 32 --lr 0.001 --epoch 50 --batch_size 4096 --eval_batch_size 8192 --gpu 0

python src/main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food --emb_size 32 --n_layers 2 --lr 0.001 --epoch 50 --early_stop 10 --batch_size 4096 --eval_batch_size 8192 --gpu 0

python src/main.py --model_name SimRec --dataset Grocery_and_Gourmet_Food --emb_size 32 --teacher_path  "E:\HOMEWORK\大三上\机器学习\理论课大作业\ReChorus-master\model\LightGCN\LightGCN__Grocery_and_Gourmet_Food__0__lr=0.001__l2=0__emb_size=32__n_layers=2__batch_size=4096.pt"  --lambda_kd 0.3 --temp_soft 1.0 --lr 0.001 --epoch 50 --early_stop 10 --batch_size 4096 --eval_batch_size 8192 --gpu 0

### 修改的代码说明

1.ReChorus-master\src\models\BaseModel.py    

将neg_items = self.data['neg_items'][index]

改为
```
if self.phase != 'train':
    if 'neg_items' in self.data:
        neg_items = self.data['neg_items'][index]
    else:
        # dev/test 阶段自动随机采样负样本
        user = self.data['user_id'][index]
        pos_item = self.data['item_id'][index]

        all_items = np.arange(1, self.corpus.n_items)
        neg_items = np.setdiff1d(all_items, [pos_item])

        # 取前 num_neg 个（或随机）
        neg_items = np.random.choice(
            neg_items,
            size=self.model.num_neg,
            replace=False
        )
else:
    neg_items = self.data['neg_items'][index]
```
2.ReChorus-master\src\models\general\SimRec.py

模型代码

## 消融实验和超参实验
### 源码

在SimRec分支

### 运行指令

在框架中运行可能要自行调整main.py文件地址）

python Main.py --epoch 50 --data yelp --teacher_model lightgcn_yelp --save_path full_simrec_yelp

python Main.py --epoch 50 --data yelp --teacher_model lightgcn_yelp --save_path ablation_1_yelp --softreg 0 

python Main.py --epoch 50 --data yelp --teacher_model lightgcn_yelp --save_path ablation_2_yelp --cdreg 0

python Main.py --epoch 50 --data yelp --teacher_model lightgcn_yelp --save_path ablation_3_yelp --screg 0

python Main.py --epoch 50 --data gowalla --teacher_model lightgcn_gowalla --save_path full_simrec_gowalla

python Main.py --epoch 50 --data gowalla --teacher_model lightgcn_gowalla --save_path ablation_1_gowalla --softreg 0

python Main.py --epoch 50 --data gowalla --teacher_model lightgcn_gowalla --save_path ablation_2_gowalla --cdreg 0

python Main.py --epoch 50 --data gowalla --teacher_model lightgcn_gowalla --save_path ablation_3_gowalla --screg 0

************超参实验运行指令（在框架中运行可能要自行调整main.py文件地址）*************

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_softreg_1e-3_gowalla  --softreg 0.001

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_softreg_1e-2_gowalla  --softreg 0.01

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_softreg_1e-1_gowalla  --softreg 0.1

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_softreg_1_gowalla  --softreg 1

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_softreg_1e1_gowalla  --softreg 10

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_cdreg_1e-4_gowalla  --cdreg 0.0001

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_cdreg_1e-3_gowalla  --cdreg 0.001

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_cdreg_1e-2_gowalla  --cdreg 0.01

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_cdreg_1e-1_gowalla  --cdreg 0.1

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_cdreg_1_gowalla  --cdreg 1

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_screg_0.5_gowalla  --screg 0.5

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_screg_1_gowalla  --screg 1

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_screg_5_gowalla  --screg 5

python Main.py --epoch 20 --data gowalla --teacher_model lightgcn_gowalla --save_path hyper_screg_10_gowalla  --screg 10
