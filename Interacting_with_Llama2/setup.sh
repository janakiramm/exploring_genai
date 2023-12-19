model=meta-llama/Llama-2-7b-chat-hf
volume=$PWD/data
token=<HUGGING_FACE_HUB_TOKEN>

docker run -d \
	--name hf-tgi \
	--runtime=nvidia \
	--gpus all \
	-e HUGGING_FACE_HUB_TOKEN=$token \
	-p 8080:80 \
	-v $volume:/data \
	ghcr.io/huggingface/text-generation-inference:1.1.0 \
	--model-id $model \
	--max-input-length 2048 \
	--max-total-tokens 4096



