# machinelearning_simrec
the homework of machinelearning

************对比实验运行指令*************
python src/main.py --model_name BPRMF --dataset MovieLens_1M --emb_size 32 --lr 1e-3 --l2 1e-6 --test_all 1 --epochs 100 --batch_size 512 --early_stop 20 --eval_batch_size 16 --num_workers 0
python src/main.py --model_name NeuMF --dataset MovieLens_1M --emb_size 32 --lr 1e-3 --l2 1e-6 --test_all 1 --epochs 100 --batch_size 256 --early_stop 20 --eval_batch_size 16 --layers "[32]" --num_workers 0
python src/main.py --model_name SimRec --dataset MovieLens_1M --emb_size 32 --lr 1e-3 --l2 1e-6 --test_all 1 --epochs 100 --batch_size 256 --early_stop 20 --eval_batch_size 16 --teacher_num 2 --lambda_kd 0.5 --num_workers 0
