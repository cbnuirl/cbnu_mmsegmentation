_base_ = './ocrnet_hr18_512x1024_40k_cityscapes.py'
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18_small',
    backbone=dict(
        extra=dict(
            stage1=dict(num_blocks=(2, )),
            stage2=dict(num_blocks=(2, 2)),
            stage3=dict(num_modules=3, num_blocks=(2, 2, 2)),
            stage4=dict(num_modules=2, num_blocks=(2, 2, 2, 2)))))

data_root = 'data/morai_cityscapes_mix/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train'),
    val=dict(
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val'),
    test=dict(
        data_root=data_root,
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val'))
