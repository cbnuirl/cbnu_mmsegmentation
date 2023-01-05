# Semantic Segmentation with HRNet + OCR on MORAI dataset

This repository is based on [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) and [mmdetection](https://github.com/open-mmlab/mmdetection). All configurations and codes were revised for MORAI dataset.

## Results and Models

### Swin-L + FPN + Cascade R-CNN

| Dataset | Lr Schd(Epoch) | box AP(vehicle) | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Real | 36 | 85.8 | [config](configs/swin/cascade_mask_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_real.py) | [log](https://drive.google.com/file/d/15eQNQVo6GkVEQruNnyGUa1vfqzQ_HPZX/view?usp=share_link) | [model] |
| Daegu | 36 | 68.3 | [config](configs/swin/cascade_mask_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_daegu.py) | [log](https://drive.google.com/file/d/1tYdIgFhjfbFgy4Hkm6ARBoFODyri2QdL/view?usp=share_link) | [model] |
| Sejong BRT 1 | 36 | 70.5 | [config](configs/swin/cascade_mask_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_sejong_1.py) | [log](https://drive.google.com/file/d/1w5hY-Gnq1xZFZPMTlxndfBPVBMTM31jX/view?usp=share_link) | [model] |
| Sangam Edge | 36 | 71.1 | [config](configs/swin/cascade_mask_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_sangam_edge.py) | [log](https://drive.google.com/file/d/1ZcHoSe4LyJZKgbCYGTfwSdYx8UtL9OTt/view?usp=share_link) | [model] |
| Sejong BRT 1 Edge | 36 | 69.6 | [config](configs/swin/cascade_mask_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_sejong_1_edge.py) | [log](https://drive.google.com/file/d/1V_931i0cTPIEAzVbcclSJAqVWg_fREvB/view?usp=share_link) | [model] |

Real:

![image](https://user-images.githubusercontent.com/121915405/210733701-69fc0980-25f1-46ee-8a09-7fe45d8f00d5.png)

Daegu:

![image](https://user-images.githubusercontent.com/121915405/210734027-8063530b-f681-40a0-bc2a-e1c384e80c19.png)

Sejong BRT 1:

![image](https://user-images.githubusercontent.com/121915405/210734170-6bc45f23-a04c-450e-b26f-18c3bb1c695a.png)

Sangam Edge:

![image](https://user-images.githubusercontent.com/121915405/210734297-974dc6cc-3a1b-48f3-bd17-21f5a32a051f.png)

Sejong BRT 1 Edge:

![image](https://user-images.githubusercontent.com/121915405/210734349-81347c31-8768-41ab-9f43-f871e9993953.png)

### Mixed Models(10% real + 90% synthetic)

| Dataset | Lr Schd(Epoch) | Real test-set box AP(vehicle) | config | log | model |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Sejong BRT 1 | 36 | 71.3 | [config](configs/swin/cascade_mask_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_sejong_1_mix.py) | [log](https://drive.google.com/file/d/1GN7tjMUQcrCaEuRTJgSGp1AUAxbJCUvh/view?usp=share_link) | [model] |
| Sejong BRT 1 Edge | 36 | 64.8 | [config](configs/swin/cascade_mask_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_morai_sejong_1_edge_mix.py) | [log](https://drive.google.com/file/d/1CgcZwRIv16wBCu4D_ghDLHmwjA_XHR38/view?usp=share_link) | [model] |

Sejong BRT 1:

![image](https://user-images.githubusercontent.com/121915405/210734549-9708315c-b365-4ef5-837c-e4f2d1ae4f15.png)

Sejong BRT 1 Edge:

![image](https://user-images.githubusercontent.com/121915405/210734588-642aca3f-6813-4dc1-abff-4985e7d61db9.png)

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
