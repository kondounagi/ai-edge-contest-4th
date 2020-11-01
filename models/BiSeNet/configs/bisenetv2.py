
## bisenetv2 なんか、たくさん出てくるから仕方ないけど、あんまり使いたくないんだよなあ
cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=4,
    lr_start = 5e-3,
    weight_decay=5e-5,
    warmup_iters = 5000,
    max_iter = 200000,
    im_root='./datasets/signate',
    train_im_anns='./datasets/signate/signate_train.txt',
    val_im_anns='./datasets/signate/signate_val.txt',
    scales=[0.25, 2.],
    #cropsize=[512, 1024],
    cropsize=[32, 64],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='./res',
)
