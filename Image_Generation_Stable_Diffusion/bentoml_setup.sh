pip install bentoml diffusers transformers accelerate
mkdir sd && cd sd

python3 fetch_sd.py
bentoml models list

ufw allow 3000
bentoml serve service:svc