# dataset settings
data_source = 'BigearthNet'
dataset_type = 'SingleViewDataset_rgbn'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406, 0.485], std=[0.229, 0.224, 0.225, 0.485])
train_pipeline = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.2, 1.0), interpolation=3),
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/NAS6/Members/dengzhuo/bigearthdata/npy_rgb_filelist/train_images.npy',
            ann_file='/NAS6/Members/dengzhuo/bigearthdata/npy_rgb_filelist/train_images.npy',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch))
