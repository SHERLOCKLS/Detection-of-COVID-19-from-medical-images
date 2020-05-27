class DefaultConfigs(object):
    train_data = './data/CT/train' # or X
    test_data = './data/CT/test'
    val_data = './data/CT/valid'
    weights = './data/checkpoints/'
    best_model = weights+'best_model/'
    logs = './data/logs/'
    gpus = '0,1'
    
    epoches = 100
    batch_size = 32
    img_height = 512
    img_width = 512
    channel_nums = 300
    seed = 666
    lr=1e-2
    lr_decay = 1e-4
    weight_decay = 1e-4
    
config = DefaultConfigs