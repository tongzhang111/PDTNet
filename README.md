## Implementation

### Dataset Preparation
The used datasets are placed in `data` folder with the following structure.
```
data
|_ vidstg
|  |_ videos
|  |  |_ [video name 0].mp4
|  |  |_ [video name 1].mp4
|  |  |_ ...
|  |_ vstg_annos
|  |  |_ train.json
|  |  |_ ...
|  |_ sent_annos
|  |  |_ train_annotations.json
|  |  |_ ...
|  |_ data_cache
|  |  |_ ...
|_ hc-stvg2
|  |_ v2_video
|  |  |_ [video name 0].mp4
|  |  |_ [video name 1].mp4
|  |  |_ ...
|  |_ annos
|  |  |_ hcstvg_v2
|  |  |  |_ train.json
|  |  |  |_ test.json
|  |  data_cache
|  |  |_ ...
|_ hc-stvg
|  |_ v1_video
|  |  |_ [video name 0].mp4
|  |  |_ [video name 1].mp4
|  |  |_ ...
|  |_ annos
|  |  |_ hcstvg_v1
|  |  |  |_ train.json
|  |  |  |_ test.json
|  |  data_cache
|  |  |_ ...
```

The download link for the above-mentioned document is as follows:

**hc-stvg**: [v1_video](https://intxyz-my.sharepoint.com/:f:/g/personal/zongheng_picdataset_com/EgIzBzuHYPtItBIqIq5hNrsBBE9cnhJDWjXuorxXMhMZGQ?e=qvsBjE), [annos](https://intxyz-my.sharepoint.com/:f:/g/personal/zongheng_picdataset_com/EgIzBzuHYPtItBIqIq5hNrsBBE9cnhJDWjXuorxXMhMZGQ?e=qvsBjE), [data_cache](https://disk.pku.edu.cn/link/AA66258EA52A1E435B815C4BC10E88925D)

**hc-stvg2**: [v2_video](https://intxyz-my.sharepoint.com/:f:/g/personal/zongheng_picdataset_com/ErqA01jikPZKnudZe6-Za9MBe17XXAxJr9ODn65Z2qGKkw?e=7vKw1U), [annos](https://intxyz-my.sharepoint.com/:f:/g/personal/zongheng_picdataset_com/ErqA01jikPZKnudZe6-Za9MBe17XXAxJr9ODn65Z2qGKkw?e=7vKw1U), [data_cache]()

**vidstg**: [videos](https://disk.pku.edu.cn/link/AA93DEAF3BBC694E52ACC5A23A9DC3D03B), [vstg_annos](https://disk.pku.edu.cn/link/AA9BD598C845DC43A4B6A0D35268724E4B), [sent_annos](https://github.com/Guaranteer/VidSTG-Dataset), [data_cache](https://disk.pku.edu.cn/link/AAA0FA082DEB3D47FCA92F3BF8775EA3BC) 



### Model Preparation
The used datasets are placed in `model_zoo` folder

[ResNet-101](https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1), 
[VidSwin-T](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth),
[roberta-base](https://huggingface.co/FacebookAI/roberta-base)

### Requirements
The code has been tested and verified using PyTorch 2.0.1 and CUDA 11.7. However, compatibility with other versions is also likely. To install the necessary requirements, please use the commands provided below:

```shell
pip3 install -r requirements.txt
apt install ffmpeg -y
```

### Training
Please utilize the script provided below:
```shell
# run for HC-STVG
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train_net.py \
    --config-file "experiments/hcstvg.yaml" \
    INPUT.RESOLUTION 420 \
    OUTPUT_DIR output/hcstvg \
    TENSORBOARD_DIR output/hcstvg

# run for HC-STVG2
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train_net.py \
    --config-file "experiments/hcstvg2.yaml" \
    INPUT.RESOLUTION 420 \
    OUTPUT_DIR output/hcstvg2 \
    TENSORBOARD_DIR output/hcstvg2

# run for VidSTG
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train_net.py \
    --config-file "experiments/vidstg.yaml" \
    INPUT.RESOLUTION 420 \
    OUTPUT_DIR output/vidstg \
    TENSORBOARD_DIR output/vidstg
```
For additional training options, such as utilizing different hyper-parameters, please adjust the configurations as needed:
`experiments/hcstvg.yaml`, `experiments/hcstvg2.yaml` and `experiments/vidstg.yaml`.

### Evaluation
Please utilize the script provided below:
```shell
# run for HC-STVG
python3 -m torch.distributed.launch \
 --nproc_per_node=8 \
 scripts/test_net.py \
 --config-file "experiments/hcstvg.yaml" \
 INPUT.RESOLUTION 420 \
 MODEL.WEIGHT [Pretrained Model Weights] \
 OUTPUT_DIR output/hcstvg
 
# run for HC-STVG2
python3 -m torch.distributed.launch \
 --nproc_per_node=8 \
 scripts/test_net.py \
 --config-file "experiments/hcstvg2.yaml" \
 INPUT.RESOLUTION 420 \
 MODEL.WEIGHT [Pretrained Model Weights] \
 OUTPUT_DIR output/hcstvg2

# run for VidSTG
python3 -m torch.distributed.launch \
 --nproc_per_node=8 \
 scripts/test_net.py \
 --config-file "experiments/vidstg.yaml" \
 INPUT.RESOLUTION 420 \
 MODEL.WEIGHT [Pretrained Model Weights] \
 OUTPUT_DIR output/vidstg
```

### Pretrained Model Weights
We provide our trained checkpoints for results reproducibility.



## Experiments
🎏 CG-STVG achieves state-of-the-art performance on three challenging benchmarks, including [**HCSTVG-v1**](https://github.com/tzhhhh123/HC-STVG), [**HCSTVG-v2**](https://github.com/tzhhhh123/HC-STVG), and [**VidSTG**](https://github.com/Guaranteer/VidSTG-Dataset), as shown below. Note that, the baseline is our CG-STVG without context generation and refinement.





## Acknowledgement
This repo is partly based on the open-source release from [STCAT](https://github.com/jy0205/STCAT) and the evaluation metric implementation is borrowed from [TubeDETR](https://github.com/antoyang/TubeDETR) for a fair comparison.


