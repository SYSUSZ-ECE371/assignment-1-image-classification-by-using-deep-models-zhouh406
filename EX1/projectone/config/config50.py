_base_ = [
    'D:/DeskTop/mmpretrain/projects/homework/_base_/models/resnet18.py',
    'D:/DeskTop/mmpretrain/projects/homework/_base_/datasets/imagenet_bs32.py',
    'D:/DeskTop/mmpretrain/projects/homework/_base_/schedules/imagenet_bs256.py',
    'D:/DeskTop/mmpretrain/projects/homework/_base_/default_runtime.py'
]



# Switch to a more powerful backbone (ResNet50 instead of ResNet18)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,  # Upgrade to ResNet50 from ResNet18
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,  # Freeze first stage only to preserve low-level features
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=2048,  # Update for ResNet50 (512 for ResNet18, 2048 for ResNet50)
        loss=dict(
            type='LabelSmoothLoss',  # Label smoothing instead of standard CE Loss
            label_smooth_val=0.1,
            mode='original'),
        cal_acc=True,
        topk=(1, )
    )
)

# Enhanced data preprocessor with mixup and cutmix
data_preprocessor = dict(
    type='ClsDataPreprocessor',
    num_classes=5,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
    # Apply mixup and cutmix for better generalization

)

# Enhanced data pipeline with stronger augmentations
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # Additional augmentations for better generalization
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3),
    dict(
        type='ColorJitter',
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1),
    dict(type='RandomRotate', prob=0.5, angle=15),
    dict(
        type='RandAugment',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

# Dataset settings with proper class balancing
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',  # Use your custom dataset
        data_root='path/to/flower_dataset',  # Update with your dataset path
        pipeline=train_pipeline,
        # Implement class balancing for handling any imbalance
        balancing_oversample_cfg=dict(
            type='ClassBalancing',
            oversample_thr=0.1),
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type='CustomDataset',  # Use your custom dataset
        data_root='path/to/flower_dataset',  # Update with your dataset path
        pipeline=test_pipeline,
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

val_evaluator = dict(type='Accuracy', topk=(1,))
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# Improved training strategy
# 1. LR strategy: Cosine annealing with warmup
# 2. Longer training with appropriate early stopping
paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.backbone.conv1': dict(lr_mult=0.1),
        '.backbone.norm1': dict(lr_mult=0.1),
    })

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',  # Use AdamW instead of SGD
        lr=0.001,  # Lower base learning rate for AdamW
        weight_decay=0.05,  # Increased weight decay for regularization
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=paramwise_cfg,
    clip_grad=dict(max_norm=5.0),  # Gradient clipping to prevent exploding gradients
)

# Cosine annealing learning rate schedule with warm-up
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),  # 5-epoch warm-up
    dict(
        type='CosineAnnealingLR',
        T_max=95,  # Total epochs - warmup epochs
        by_epoch=True,
        begin=5,
        end=100,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

# Run for more epochs (100 instead of standard 90/100)
train_cfg = dict(
    by_epoch=True,
    max_epochs=100,
    val_interval=1)

# Use exponential moving average (EMA) for better stability and convergence
custom_hooks = [
    dict(type='EMAHook', momentum=0.0001, priority='ABOVE_NORMAL'),
]

# Use more advanced checkpointing strategy
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        save_best='auto',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='val/accuracy',
        patience=15,
        rule='greater'),
)

# Use automatic mixed precision training to speed up training
amp = True  # Automatic mixed precision

# Set work directory for saving checkpoints
work_dir = 'D:\DeskTop\mmpretrain\work_dirs\config50'
