# machinelearning_simrec
the homework of machinelearning

************对比实验运行指令*************
python src/main.py --model_name BPRMF --dataset MovieLens_1M --emb_size 32 --lr 1e-3 --l2 1e-6 --test_all 1 --epochs 100 --batch_size 512 --early_stop 20 --eval_batch_size 16 --num_workers 0
python src/main.py --model_name NeuMF --dataset MovieLens_1M --emb_size 32 --lr 1e-3 --l2 1e-6 --test_all 1 --epochs 100 --batch_size 256 --early_stop 20 --eval_batch_size 16 --layers "[32]" --num_workers 0
python src/main.py --model_name SimRec --dataset MovieLens_1M --emb_size 32 --lr 1e-3 --l2 1e-6 --test_all 1 --epochs 100 --batch_size 256 --early_stop 20 --eval_batch_size 16 --teacher_num 2 --lambda_kd 0.5 --num_workers 0

************消融实验运行指令（在框架中运行可能要自行调整main.py文件地址）*************
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
