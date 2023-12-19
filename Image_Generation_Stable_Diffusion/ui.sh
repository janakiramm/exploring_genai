adduser demo
echo "demo       ALL=(ALL:ALL) NOPASSWD:ALL" | sudo tee -a /etc/sudoers
su demo

sudo apt get update
sudo apt install -y wget git python3 python3-venv libgl1 libglib2.0-0

mkdir sd
wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
chmod +x ./webui.sh
sudo ufw allow 7860
./webui.sh --listen
