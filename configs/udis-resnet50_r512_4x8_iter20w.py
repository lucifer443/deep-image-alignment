model= dict(
            type='UDIS_H_Predictor',
            feature_extractor=dict( type='ResNet',
                                    depth=50,
                                    num_stages=4,
                                    out_indices=(1, 2, 3),
                                    frozen_stages=1,
                                    norm_cfg=dict(type='BN', requires_grad=True),
                                    norm_eval=True,
                                    style='pytorch',
                                    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
            H_decoder=dict(type='UDISDecoder', 
                           feat_size=[(16, 16), (32, 32), (64, 64)],
                           strides=[32, 16, 8],
                           search_range=[16, 8, 4],
                           )
        )

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

train_pipeline = [
    dict(type='LoadImagePairFromFile', to_float32=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='BeforeDataAugment'),
    dict(type='PairPhotoMetricDistortion'),
    dict(type='ImageToTensor', keys=['img1', 'img2', 'raw_img1', 'raw_img2']),
    dict(type='Collect', keys=['img1', 'img2', 'raw_img1', 'raw_img2'],
                         meta_keys=['filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'scale_factor'])
]

test_pipeline = [
    dict(type='LoadImagePairFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='ImageToTensor', keys=['img1', 'img2']),
    dict(type='Collect', keys=['img1', 'img2'],
                         meta_keys=['filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'scale_factor'])
]

dataset_type = 'UDISDataset'
data_root = 'data/udis-d/'

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'training_infos.pkl',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'testing_infos.pkl',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'testing_infos.pkl',
        img_prefix=data_root,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    by_epoch=False,
    min_lr=1e-6)
runner = dict(type='IterBasedRunner', max_iters=200000)

checkpoint_config = dict(by_epoch=False, interval=10000)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
