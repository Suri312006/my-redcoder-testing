# pip install cython
# pip install faiss-cpu
# pip install filelock
# pip install numpy
# pip install regex
# pip install torch
# pip install transformers
# pip install tqdm
# pip install wget
# pip install spacy


export DEVICES=0
export NUM_DEVICES=1
export PRETRAINED_MODEL_PATH=./data/graphcodebert-base
export TRAIN_FILE_PATH=/pvc-surendra/REDCODER/SCODE-R/data/java/final/jsonl/train/java_train_0.jsonl
export DEV_FILE_PATH=/pvc-surendra/REDCODER/SCODE-R/data/java/final/jsonl/train/java_train_0.jsonl
# export TRAIN_FILE_PATH=/pvc-surendra/REDCODER/SCODE-R/data/java/final/jsonl
# export DEV_FILE_PATH=/pvc-surendra/REDCODER/SCODE-R/data/java/final/jsonl
export OUTPUT_DIR=./output
export CUDA_VISIBLE_DEVICES=0

#python -m torch.distributed.launch      --nproc_per_node=${NUM_DEVICES} train_dense_encoder.py      --max_grad_norm 2.0 --encoder_model_type hf_roberta --pretrained_model_cfg ${PRETRAINED_MODEL_PATH} --eval_per_epoch 1 --seed 12345 --sequence_length 256 --warmup_steps 1237 --batch_size 8 --train_file ${TRAIN_FILE_PATH} --dev_file ${DEV_FILE_PATH} --output_dir ${OUTPUT_DIR} --learning_rate 2e-5 --num_train_epochs 15 --dev_batch_size 64 --val_av_rank_start_epoch 0 --fp16 
torchrun --nproc_per_node=${NUM_DEVICES} train_dense_encoder.py      --max_grad_norm 2.0 --encoder_model_type hf_roberta --pretrained_model_cfg ${PRETRAINED_MODEL_PATH} --eval_per_epoch 1 --seed 12345 --sequence_length 256 --warmup_steps 1237 --batch_size 8 --train_file ${TRAIN_FILE_PATH} --dev_file ${DEV_FILE_PATH} --output_dir ${OUTPUT_DIR} --learning_rate 2e-5 --num_train_epochs 15 --dev_batch_size 64 --val_av_rank_start_epoch 0 --fp16 




