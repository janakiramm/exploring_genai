pip install huggingface_hub
pip install autotrain-advanced

autotrain setup

huggingface-cli login

export model=meta-llama/Llama-2-7b-chat-hf
export data_path=smangrul/ad-copy-generation
export text_column=content

export token=<HUGGING_FACE_TOKEN>
export project_name=<PROJECT_NAME>
export repoid=<HF_USER/MODEL>

autotrain llm \
  --train \
  --project_name $project_name  \
  --model $model \
  --data_path $data_path \
  --text_column $text_column \
  --use_peft \
  --use_int4 \
  --learning_rate 2e-4 \
  --train_batch_size 2 \
  --num_train_epochs 1 \
  --trainer sft \
  --block_size 1024 \
  --lora_r 16 \
  --lora_alpha 32 \
  --merge-adapter \
  --lora_dropout 0.045 \
  --push-to-hub \
  --token $token \
  --repo-id $repoid
