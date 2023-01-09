# Environment setting

## Install Docker

### Install with script

```
sudo wget -qO- https://get.docker.com/ | sh
```
or
```
sudo curl -fsSL https://get.docker.com/ | sh
```

## Git Clone

```
git clone https://github.com/cbnuirl/cbnu_mmsegmentation.git
cd cbnu_mmsegmentation
```

## Docker Image

### Compressed file(if exists)

```
sudo docker load -i mmseg_cbnu.tar
```

### dockerfile

```
sudo docker build –t cbnuirl/mmseg_cbnu:1.0 docker/
```

## Docker Container

```
sudo docker run (--gpus all) --shm-size=8g –it –v —name {CONTAINER_NAME} \
{WORK_DIR}:/mmsegmentation cbnuirl/mmseg_cbnu:1.0
```

--gpus all for multi-GPU. Exclude when using single-GPU.

If you want to change into container bash:
```
sudo docker start {CONTAINER_NAME}
sudo docker attach {CONTAINER_NAME}
```

# Data Conversion

Change into Cityscapes style custom format like:
```
data/
	└(DATA_NAME)
		├gtFine
		│	├train/
		│	│	└XX_XX_TXWX_XX_XXX_REXX_XXX_labelIds.png
		│	│	-> Grayscaled label image
		│	└val/
		│		└XX_XX_TXWX_XX_XXX_REXX_XXX_labelIds.png
		└leftImg8bit
			├train/
			│	└XX_XX_TXWX_XX_XXX_REXX_XXX.png
			│	-> Original image
			└val/
				└XX_XX_TXWX_XX_XXX_REXX_XXX.png
```

Grayscaled label image looks like:

![UR_SE_T1W1_U1_N01_RE01_001_labelIds](https://user-images.githubusercontent.com/121915405/210936666-8604bf25-4f7c-4c21-ad50-34e28559d0a8.png)

Prepare data like:
```
data/
	└(DATA_NAME)
		├leftImg8bit
		│	├train/
		│	│	└XX_XX_TXWX_XX_XXX_REXX_XXX.png
		│	│	-> Original image
		│	└val/
		│		└XX_XX_TXWX_XX_XXX_REXX_XXX.png
		└label/
			└XX_XX_TXWX_XX_XXX_REXX_XXX.png
			->MORAI label images that have RGB value
```

MORAI label image looks like:

![UR_SE_T1W1_U1_N01_RE01_001](https://user-images.githubusercontent.com/121915405/210936561-74a1865f-7525-44d2-bb33-584814245f3f.png)

Change color_map of tools/convert_datasets/morai.py if categories or segmentation colors are different:
```
…
color_map = [
    [255, 90, 241],     # vehicle
    [66, 7, 158],       # bus
    [0, 243, 64],       # truck
    [248, 158, 235],    # policeCar
    [211, 222, 241],    # ambulance
    [255, 255, 255],    # schoolBus
    [112, 32, 48],      # otherCar
    [255, 255, 255],    # motorcycle
    [248, 171, 255],    # bicycle
    [255, 255, 255],    # twoWheeler
    [255, 0, 233],      # pedestrian
    [255, 255, 255],    # rider
    [9, 161, 181],      # freespace
    [80, 180, 98],      # curb
    [118, 78, 176],     # sidewalk
    [255, 255, 255],    # crossWalk
    [0, 207, 255],      # safetyZone
    [134, 59, 141],     # speedBump
    [248, 28, 81],      # roadMark
    [255, 188, 123],    # whiteLane
    [255, 100, 0],      # yellowLane
    [152, 255, 141],    # blueLane
    [255, 255, 255],    # redLane
    [106, 163, 145],    # stopLane
    [241, 90, 41],      # trafficSign
    [0, 233, 197],      # trafficlight
    [0, 182, 255],      # constructionGuide
    [136, 153, 189],    # trafficDrum
    [255, 255, 255],    # rubberCone
    [255, 255, 255],    # warningTriangle
    [247, 234, 110],    # fence
    [255, 255, 255],    # egoVehicle
    [210, 229, 168]     # background
]
…
```

[255, 255, 255]is undefined colors of classes.

TrafficDrum was [136, 153, 179] but grayscaled color is same as safetyZone, so temporary modified to [136, 153, 189] for next procedure.

Change mmseg/datasets/custom.py CLASSES, PALETTE:
```
CLASSES = ['vehicle', 'bus', 'truck', 'policeCar', 'ambulance',
		'schoolBus', 'otherCar', 'motorcycle', 'bicycle', 'twoWheeler',
		'pedestrian', 'rider', 'freespace', 'curb', 'sidewalk',
		'crossWalk', 'safetyZone', 'speedBump', 'roadMark', 'whiteLane',
		'yellowLane', 'blueLane', 'redLane', 'stopLane', 'trafficSign',
		'trafficLight', 'constructionGuide', 'trafficDrum', 'rubberCone', 'warningTriangle',
		'fence', 'egoVehicle', 'background']

PALETTE = [[255, 90, 241], [66, 7, 158], [0, 243, 64], [248, 158, 235], [211, 222, 241],
		[144, 64, 0], [112, 32, 48], [176, 0, 96], [248, 171, 255], [176, 0, 128],
		[255, 0, 233], [160, 0, 0], [9, 161, 181], [80, 180, 98], [118, 78, 176],
		[48, 16, 0], [0, 207, 255], [134, 59, 141], [248, 28, 81], [255, 188, 123],
		[255, 100, 0], [152, 255, 141], [32, 48, 0], [106, 163, 145], [241, 90, 41],
		[0, 233, 197], [0, 182, 255], [136, 153, 179], [176, 0, 0], [176, 0, 16],
		[247, 234, 110], [0, 0, 0], [210, 229, 168]]
```

Then, run:
```
pip install tqdm numpy opencv-python # If not installed
```
```
python tools/convert_datasets/morai.py --data_path {DATA_PATH}
```

{DATA_PATH} would be like data/(DATA_NAME). label folder can be deleted.

# Change Configuration File

Create new configuration file for new data. Location is configs/ocrnet/.
Save like 'ocrnet_hr48_512x1024_160k_{DATA_NAME}.py'
Modify content:
```
…
data_root = ‘data/{DATA_NAME}/’
…
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))
```

For mixed training:
```
…
dataset_A_train=dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='synthetic/leftImg8bit/train',
    ann_dir='synthetic/gtFine/train',
    pipeline=train_pipeline)

dataset_B_train=dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='real/leftImg8bit/train',
    ann_dir='real/gtFine/train',
    pipeline=train_pipeline)

dataset_B_test=dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='real/leftImg8bit/val',
    ann_dir='real/gtFine/val',
    pipeline=test_pipeline)

dataset_B_val=dict(
    type=dataset_type,
    data_root=data_root,
    img_dir='real/leftImg8bit/val',
    ann_dir='real/gtFine/val',
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
	dataset_A_train,
	dataset_B_train
    ],
    val=dataset_B_val,
    test=dataset_B_val)
```

Mixed training data should be distributed like:
```
data
	└(DATA_NAME)/
		├real/
		│	├gtFine/
		│	│	├train/
		│	│	└val/
		│	└leftImg8bit/
		│		├train/
		│		└val/
		└synthetic/
			├gtFine/
			│	├train/
			│	└(val/)
			└leftImg8bit/
				├train/
				└(val/)
```
Validation set that will not be used can be removed.

**NOTE**: Original image format(.png or .jpg) and label image format(labelImg.png) should be same between real and synthetic data.

If you want to use pre-trained model to continue training or fine-tuning, add configuration file:
```
_base_ = …
norm_cfg = …
load_from = “(PRETRAINED_MODEL)”
…
```
PRETRAINED_MODEL will be like "checkpoints/ocrnet…morai_daegu.pth"
