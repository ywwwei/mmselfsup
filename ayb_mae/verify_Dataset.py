# verify the data_source and dataloader
import mmselfsup.datasets.single_view_rgbn as single_view_rgbn

# dataset settings
# data_source = 'BigearthNet'

data_source=dict(
            type='BigearthNet',
            data_prefix='/NAS6/Members/dengzhuo/bigearthdata/npy_rgb_filelist/train_images.npy',
            ann_file='/NAS6/Members/dengzhuo/bigearthdata/npy_rgb_filelist/train_images.npy',
        )
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

dataset = single_view_rgbn.SingleViewDataset_rgbn(data_source=data_source, pipeline=train_pipeline, prefetch=prefetch )
print("Dataset is ready!")