# Semantic Segmentation with HRNet + OCR on MORAI dataset

This repository is based on [HRNet + OCR](https://github.com/HRNet/HRNet-Semantic-Segmentation) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation). All configurations and codes were revised for MORAI dataset.

## Results and Models

### HRNet + OCR

Class accuracy is IoU(Intersection over Union).

| Dataset | Iter | vehicle | bus | truck | freespace | roadMark | whiteLane | yellowLane | trafficSign | fence | background | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Real | 160k | 93.29 | 89.92 | 87.72 | 94.94 | 61.22 | 87.81 | 79.86 | 66.3 | 91.08 | 97.85 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_real.py) | [log](https://drive.google.com/file/d/1mlQ0PNpR6onGbAYglsb5qQkZ8w7acRdK/view?usp=share_link) | [model] |
| Daegu | 160k | 93.16 | 71.36 | 87.35 | 98.03 | 82.69 | 83.64 | 84.3 | 71.6 | 83.87 | 99.58 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_daegu.py) | [log](https://drive.google.com/file/d/1t1xb1mEck7aoTTl-186CcXwSIMzu4Y1S/view?usp=share_link) | [model] |
| Sejong BRT 1 | 160k | 55.83 | 75.51 | 53.52 | 98.67 | 66.46 | 82.2 | 63.77 | 72.14 | X | 99.83 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sejong_1.py) | [log](https://drive.google.com/file/d/13Qjd1q-fx2yS7KiZMVKHh_pdwiv-mjMW/view?usp=share_link) | [model] |
| Sangam Edge | 160k | 84.57 | 0.0 | 0.0 | 97.77 | 82.93 | 79.45 | 84.59 | 91.56 | 92.47 | 99.03 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sangam_edge.py) | [log](https://drive.google.com/file/d/1mHIskUGH0-CAmaZo3Is1HtZrc2k0NOyC/view?usp=share_link) | [model] |
| Sejong BRT 1 Edge | 160k | 53.91 | 0.0 | 0.0 | 98.4 | X | 92.01 | 35.98 | 0.0 | X | 99.56 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sejong_1_edge.py) | [log](https://drive.google.com/file/d/133zjdQnpArwUgwNzlFYfAEZwLXKxXJVc/view?usp=share_link) | [model] |

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
| Sejong BRT 1 | 160k | 86.25 | 77.94 | 62.12 | 95.82 | 49.5 | 82.73 | 74.57 | 37.21 | 67.79 | 97.68 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sejong_1_mix.py) | [log](https://drive.google.com/file/d/1yw89dC9GdyRMNJ_AjxT0m0z7y28DSj2T/view?usp=share_link) | [model] |
| Sejong BRT 1 Edge | 160k | 84.95 | 68.66 | 20.02 | 94.88 | 0.94 | 82.81 | 72.19 | 0.0 | 49.82 | 94.42 | [config](configs/ocrnet/ocrnet_hr48_512x1024_160k_morai_sejong_1_edge_mix.py) | [log](https://drive.google.com/file/d/1N1agPFDpaQZMC-7Jn0BZ8zeCHnkFhnWr/view?usp=share_link) | [model] |

Some of the classes not included in data or in small amounts.

Sejong BRT 1:

![image](https://user-images.githubusercontent.com/121915405/210933760-c69e0485-b3ca-46ff-9f21-64690ec652bb.png)

Sejong BRT 1 Edge:

![image](https://user-images.githubusercontent.com/121915405/210933781-21bbb65f-1c52-41aa-99eb-50a822c5ee11.png)

## Usage

### Installation

Please refer to [install.md](docs/install.md) for installation, dataset preparation and making configuration file.

### Testing, Demo
```
# single-gpu testing
python tools/test.py {CONFIG_FILE} {MODEL_FILE} --eval bbox \
(--show-dir {LOCATION}) \
(--options "classwise=True")

# multi-gpu testing
(CUDA_VISIBLE_DEVICES={GPU_NUM}) \
tools/dist_test.sh {CONFIG_FILE} {MODEL_FILE} {TOTAL_NUM_OF_GPU} --eval bbox \
(--show-dir {LOCATION}) \
(--options “classwise=True”)
```

--show-dir saves pictures of result, --options "classwise=True" shows average precision of all classes.
You can use --show in GUI environment.

Example:
```
python tools/test.py \
configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_daegu.py \
checkpoints/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_daegu.pth \
--eval bbox --show-dir result.bbox.daegu/ --options “classwise=True”

CUDA_VISIBLE_DEVICES=0,1,3 tools/dist_test.sh \
configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_daegu.py \
checkpoints/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_daegu.pth 3 \
--eval bbox --show-dir result.bbox.daegu/ --options “classwise=True”
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
python tools/train.py configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_daegu.py

CUDA_VISIBLE_DEVICES=0,1,3 tools/dist_train.sh \
configs/swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_daegu.py 3
```

## Citing Swin Transformer
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
