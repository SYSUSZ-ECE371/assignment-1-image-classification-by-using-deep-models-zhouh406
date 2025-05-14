_base_ = [
    'D:/DeskTop/mmpretrain/projects/homework/_base_/models/resnet18.py',
    'D:/DeskTop/mmpretrain/projects/homework/_base_/datasets/imagenet_bs32.py',
    'D:/DeskTop/mmpretrain/projects/homework/_base_/schedules/imagenet_bs256.py',
    'D:/DeskTop/mmpretrain/projects/homework/_base_/default_runtime.py'
]
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
    ))
data_preprocessor = dict(
    num_classes=5,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.001))