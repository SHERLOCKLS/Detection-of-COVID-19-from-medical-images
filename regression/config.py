class DefaultConfigs(object):
    #1.string parameters
    train_data = "./data/"
    test_data = "./"
    val_data = "no"
    model_name = "resnet"
    weights = "./checkpoints/"
    best_models = weights + "best_model/"
    submit = "./submit/"
    logs = "./logs/"
    gpus = "0"

    #2.numeric parameters
    epochs = 200
    batch_size = 20
    img_height = 64
    img_weight = 64
    num_classes = 5
    seed = 666
    lr = 1e-2
    lr_decay = 1e-4
    weight_decay = 1e-4

config = DefaultConfigs()
