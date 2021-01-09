touch ~/.no_auto_tmux
mkdir /workspace/datasets
apt-get update && apt install --assume-yes -y git wget curl htop ncdu libsndfile1 ffmpeg nano vim iputils-ping cmake
git config --global user.name "ctlaltdefeat"
git config --global user.email "ctlaltdefeat@tfwno.gf"
curl https://getcroc.schollz.com | bash
pip install Cython flake8 dill joblib black typeguard librosa gdown ipykernel
conda install gxx_linux-64
conda install pytorch-lightning -c conda-forge
pip install -U git+https://github.com/ctlaltdefeat/phonemizer.git
pip install -U git+https://github.com/ctlaltdefeat/text-frontend-tts.git
git submodule update --init
cd /workspace/efficient_tts/hifi-gan
mkdir pretrained_universal
cd pretrained_universal
gdown --id 1pAB2kQunkDuv6W5fcJiQ0CY8xcJKB22e
gdown --id 1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW
gdown --id 1O63eHZR9t1haCdRHQcEgMfMNxiOciSru
# cd /workspace/datasets
# wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# tar -xvf LJSpeech-1.1.tar.bz2