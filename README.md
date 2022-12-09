# 360-MLC: Multi-view Layout Consistency for Self-training and Hyper-parameter Tuning

<!-- ![](https://enriquesolarte.github.io/360-mlc/img/teaser.svg) -->
![](https://user-images.githubusercontent.com/67839539/205412513-39495ba4-2bf6-47d6-90c8-e948fb22576a.png)

This is the official implementation of 360-MLC, where we propose a novel approach to fine-tune and evaluate pre-trained layout estimation models on new datasets with domain shifts, and, more importantly, no ground truth is required.

For more detailed information, please refer to:
> [**360-MLC: Multi-view Layout Consistency for Self-training and Hyper-parameter Tuning**](https://arxiv.org/abs/2210.12935)          
> Bolivar Solarte, Chin-Hsuan Wu, Yueh-Cheng Liu, Yi-Hsuan Tsai, Min Sun       
> NeurIPS 2022            
> [**[Paper]**](https://arxiv.org/abs/2210.12935), [**[Project Page]**](https://enriquesolarte.github.io/360-mlc/), [**[Video]**](https://youtu.be/x4Vt32egsdU) 

## Video
[![](https://user-images.githubusercontent.com/67839539/205503534-5ea1152e-c855-4b1a-90a0-277bb2731815.png)](https://youtu.be/x4Vt32egsdU)

## Installation 

The current implementation uses `python 3.7` and `Pytorch 1.13`. For convenience, we recommend using conda, miniconda, pyenv or any environment tool to keep isolate this implementation. 

```sh
# Create and activate conda env (mlc) 
conda create -n mlc python=3.7 
conda activate mlc 

# Install Pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# Assuming $HOME directory by default
cd ~/
# clone 360-MLC (Note that we have added HorizonNet as submodule)
git clone --recurse-submodules git@github.com:EnriqueSolarte/360-mlc.git

cd ~/360-mlc

# Install python dependencies
pip install -r requirements.txt

# Install MLC library
pip install .
```

## Dataset

The dataset used for this implementation is the **MP3D-FPE** dataset released by [(Solarte et al. 2019 RA-L)](https://enriquesolarte.github.io/360-dfpe/). 
To process this dataset, please follow the next commands:

```sh 
MP3D_FPE_DIR="<MP3D_FPE dataset directory>"
python data/process_mp3d_fpe_data.py --path $MP3D_FPE_DIR

```

## How to create 360-mlc labels 

To create pseudo labels based on a pre-trained model, following the next commands: 

```sh
CKPT=zind # e.g. mp3d, zind, st3d or panos2d3d
python main_create_mlc_labels.py --ckpt $CKPT --cfg ./assets/create_mlc_labels.yaml
```
To download `ckpt` pre-trained models, you can refer to the official pre-trained models in [HorizonNet](https://github.com/sunset1995/HorizonNet/tree/4eff713f8d446c53c479d86b4d06af166b724a74#:~:text=testing%20for%20HorizonNet.-,Pretrained%20Models,-resnet50_rnn__panos2d3d.pth).

After to download a `ckpt` model, we suggest you to modify accordinally the cfg file `config/trained_models.yaml`







## Citation
> 
    @inproceedings{solarte2022mlc,
        author={Solarte, Bolivar and Wu, Chin-Hsuan and Liu, Yueh-Cheng and Tsai, Yi-Hsuan and Sun, Min},
        title={360-MLC: Multi-view Layout Consistency for Self-training and Hyper-parameter Tuning},
        booktitle={Advances in Neural Information Processing Systems},
        year={2022},
    }