# Semantic Segmentation with HRNet + OCR on MORAI dataset

This repository is based on [HRNet + OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). All configurations and codes were revised for MORAI dataset.

## Results and Models

### HRNet + OCR

Class accuracy is IoU(Intersection over Union).

| Dataset | Iter | vehicle | bus | truck | freespace | roadMark | whiteLane | yellowLane | trafficSign | fence | background | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Real | 160k | 93.29 | 89.92 | 87.72 | 94.94 | 61.22 | 87.81 | 79.86 | 66.3 | 91.08 | 97.85 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_real.py) | [log](https://drive.google.com/file/d/1mlQ0PNpR6onGbAYglsb5qQkZ8w7acRdK/view?usp=share_link) | [model](https://drive.google.com/file/d/1rNg7YLrdB3ayJ_t77kTMclRxeoW3RljC/view?usp=share_link) |
| Daegu | 160k | 93.16 | 71.36 | 87.35 | 98.03 | 82.69 | 83.64 | 84.3 | 71.6 | 83.87 | 99.58 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_daegu.py) | [log](https://drive.google.com/file/d/1t1xb1mEck7aoTTl-186CcXwSIMzu4Y1S/view?usp=share_link) | [model](https://drive.google.com/file/d/1JGE2u6HHDiVyXcvBampTPgbIJXV_a6hn/view?usp=share_link) |
| Sejong BRT 1 | 160k | 55.83 | 75.51 | 53.52 | 98.67 | 66.46 | 82.2 | 63.77 | 72.14 | X | 99.83 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sejong_1.py) | [log](https://drive.google.com/file/d/13Qjd1q-fx2yS7KiZMVKHh_pdwiv-mjMW/view?usp=share_link) | [model](https://drive.google.com/file/d/12_mVTloAo6phF1Im6h6YQ1eyHPyuPVYn/view?usp=share_link) |
| Sangam Edge | 160k | 84.57 | 0.0 | 0.0 | 97.77 | 82.93 | 79.45 | 84.59 | 91.56 | 92.47 | 99.03 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sangam_edge.py) | [log](https://drive.google.com/file/d/1mHIskUGH0-CAmaZo3Is1HtZrc2k0NOyC/view?usp=share_link) | [model](https://drive.google.com/file/d/13-FQY6z45CZEJKNptvQ7dSl-V-C7qVLl/view?usp=share_link) |
| Sejong BRT 1 Edge | 160k | 53.91 | 0.0 | 0.0 | 98.4 | X | 92.01 | 35.98 | 0.0 | X | 99.56 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sejong_1_edge.py) | [log](https://drive.google.com/file/d/133zjdQnpArwUgwNzlFYfAEZwLXKxXJVc/view?usp=share_link) | [model](https://drive.google.com/file/d/16Pz3m6UN4ioqcYkV5A-uJbgMybvy-neZ/view?usp=share_link) |

Some of the classes not included in data or in small amounts.

Real:

![image](https://user-images.githubusercontent.com/121915405/210929946-148bd854-4aa6-469e-803e-36dff218f887.png)

Daegu:

![image](https://user-images.githubusercontent.com/121915405/210929969-534794ad-6216-4c73-b7e0-b76326502481.png)

Sejong BRT 1:

![image](https://user-images.githubusercontent.com/121915405/210930001-a051fca5-4fec-45fd-b1ab-19f0b5ac1673.png)

Sangam Edge:

![image](https://user-images.githubusercontent.com/121915405/210930026-1b9c79d0-468f-4daf-ac23-3730f2c01814.png)

Sejong BRT 1 Edge:

![image](https://user-images.githubusercontent.com/121915405/210930051-93fcef68-2afe-47a9-8abc-c90ebe41bcaf.png)

### Mixed Models(10% real + 90% synthetic)

| Dataset | Iter | vehicle | bus | truck | freespace | roadMark | whiteLane | yellowLane | trafficSign | fence | background | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Daegu | 160k | 89.72 | 67.8 | 64.09 | 95.72 | 40.56 | 83.04 | 75.65 | 41.92 | 63.48 | 97.16 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_daegu_mix.py) | [log](https://drive.google.com/file/d/1VQhBiCMSraq293UvuUf_ZmVHm-ju5Ogm/view?usp=share_link) | [model] |
| Sejong BRT 1 | 160k | 86.25 | 77.94 | 62.12 | 95.82 | 49.5 | 82.73 | 74.57 | 37.21 | 67.79 | 97.68 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sejong_1_mix.py) | [log](https://drive.google.com/file/d/1yw89dC9GdyRMNJ_AjxT0m0z7y28DSj2T/view?usp=share_link) | [model](https://drive.google.com/file/d/1oFLz7R0AAVO1YsqAZOLKi9hVJ-Yfhmae/view?usp=share_link) |
| Sangam Edge | 160k | 89.55 | 68.71 | 65.34 | 95.9 | 49.17 | 84.86 | 73.56 | 38.53 | 80.18 | 97.96 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sangam_edge_mix.py) | [log](https://drive.google.com/file/d/1yfHDkCPep5LRI5Z0f2PJMjctQAXjZ1I_/view?usp=share_link) | [model] |
| Sejong BRT 1 Edge | 160k | 84.95 | 68.66 | 20.02 | 94.88 | 0.94 | 82.81 | 72.19 | 0.0 | 49.82 | 94.42 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sejong_1_edge_mix.py) | [log](https://drive.google.com/file/d/1N1agPFDpaQZMC-7Jn0BZ8zeCHnkFhnWr/view?usp=share_link) | [model](https://drive.google.com/file/d/1l-1pCv7qx9gqb9DBA4vJ86e-kHNfcACn/view?usp=share_link) |

Some of the classes not included in data or in small amounts.

Daegu:

![image](https://user-images.githubusercontent.com/121915405/211253380-c639c2e3-4cd9-4849-9e89-34436de81e53.png)

Sejong BRT 1:

![image](https://user-images.githubusercontent.com/121915405/211253399-25dfc503-1a93-44e3-acba-83f800a4f543.png)

Sangam Edge:

![image](https://user-images.githubusercontent.com/121915405/211253435-b15a432a-dcf1-4695-a8e1-e87ed6bfd038.png)

Sejong BRT 1 Edge:

![image](https://user-images.githubusercontent.com/121915405/211253467-0ab96762-02c2-4749-a285-455fa177a3ff.png)

## Usage

### Installation

Please refer to [install.md](docs/install.md) for installation, dataset preparation and making configuration file.

### Testing, Demo

**NOTE**: Change mmseg/datasets/custom.py if original image format is in .jpg

```
…
def __init__(self,
                 pipeline,
                 img_dir,
                 # img_suffix='_leftImg8bit.png',
                 img_suffix='.png', -> Modify into ‘.jpg’
…
```

```
# single-gpu testing
python tools/test.py {CONFIG_FILE} {MODEL_FILE} --eval mIoU \
(--show-dir {LOCATION})

# multi-gpu testing
(CUDA_VISIBLE_DEVICES={GPU_NUM}) \
tools/dist_test.sh {CONFIG_FILE} {MODEL_FILE} {TOTAL_NUM_OF_GPU} --eval mIoU \
(--show-dir {LOCATION})
```

--show-dir saves pictures of result.
You can use --show in GUI environment.

CUDA_VISIBLE_DEVICES can define which GPUs will be used. If not defined, they are used sequentially.

Example:
```
python tools/test.py \
configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_daegu.py \
checkpoints/ocrnet_hr48_512x1024_160k_morai_daegu.pth \
--eval mIoU --show-dir result.mIoU.daegu/

CUDA_VISIBLE_DEVICES=0,1,3 tools/dist_test.sh \
configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_daegu.py \
checkpoints/ocrnet_hr48_512x1024_160k_morai_daegu.pth 3 \
--eval mIoU --show-dir result.mIoU.daegu/
```

### Training

```
# single-gpu training
python tools/train.py {CONFIG_FILE}

# multi-gpu training
(CUDA_VISIBLE_DEVICES={GPU_NUM}) tools/dist_train.sh {CONFIG_FILE} {TOTAL_NUM_OF_GPU}
```

Example:
```
python tools/train.py configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_daegu.py

CUDA_VISIBLE_DEVICES=0,1,3 tools/dist_train.sh \
configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_daegu.py 3
```
