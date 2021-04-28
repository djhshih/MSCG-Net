conda create -y --name mscg-net python=3.7
conda activate mscg-net
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install scikit-learn ipywidgets
conda install opencv>=3.4 tensorboardx albumentations pretrainedmodels -c conda-forge
