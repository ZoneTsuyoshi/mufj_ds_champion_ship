import os, json
import numpy as np
from sklearn import metrics
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import get_train_data, LitBertForSequenceClassification, plot_confusion_matrix

def train(dirpath, debug=False):
    f = open(os.path.join(dirpath, "config.json"), "r")
    config = json.load(f)
    f.close()
    
    dirpath = dirpath + "/results"
    os.mkdir(dirpath)
    gpu = config["train"]["gpu"]
    seed = config["train"]["seed"]
    
    cuda = torch.cuda.is_available()
    print("cuda is avaiable: {}".format(cuda))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    epoch = config["train"]["epoch"]
    if debug:
        epoch = 1
        config["train"]["kfolds"] = 2
    kfolds = config["train"]["kfolds"]
    warmup_rate = config["train"]["warmup_rate"]
    using_mlm = config["train"]["using_mlm"]
    mlm_id = config["train"]["mlm_id"]
    adv_start_epoch = config["train"]["adv_start_epoch"]
    model_name = config["network"]["model_name"]
    lower_model_name = model_name.rsplit("/", 1)[1] if "/" in model_name else model_name
    gradient_clip_val = config["network"]["gradient_clipping"]
    manual_optimization = config["network"]["at"] is not None
    if manual_optimization: gradient_clip_val = None
    ckpt_name = config["test"]["ckpt"] # best / last
    
    train_loader_list, valid_loader_list, valid_labels_list, valid_indices_list, weight_list, design_dim, df = get_train_data(config, debug)
    logger = pl.loggers.CometLogger(workspace=os.environ.get("zonetsuyoshi"), save_dir=dirpath, project_name="mufj-dscs")
    flatten_config = {}
    for key in config.keys():
        flatten_config.update(config[key])
    logger.log_hyperparams(flatten_config)
    valid_probs = np.zeros(len(df))
    for i, (train_loader, valid_loader, valid_labels, valid_indices, weight) in enumerate(zip(train_loader_list, valid_loader_list, valid_labels_list, valid_indices_list, weight_list)):
        fold_id = i if valid_loader is not None else "A"
        total_steps = epoch * len(train_loader)
        warmup_steps = int(warmup_rate * total_steps) if warmup_rate < 1 else warmup_rate
        adv_start_epoch = adv_start_epoch if adv_start_epoch>=1 else int(adv_start_epoch * epoch)
        mlm_path = f"../m{mlm_id}/fold{fold_id}" if using_mlm else None
        model = LitBertForSequenceClassification(**config["network"], dirpath=dirpath, fold_id=fold_id, design_dim=design_dim, weight=weight, num_warmup_steps=warmup_steps, num_training_steps=total_steps, mlm_path=mlm_path, adv_start_epoch=adv_start_epoch)
        print(model)
        checkpoint = pl.callbacks.ModelCheckpoint(monitor=f'valid_loss{fold_id}' if valid_loader is not None else f"train_loss{fold_id}", mode='min', save_last=True, save_top_k=1, save_weights_only=True, dirpath=dirpath, filename=f"fold{fold_id}_best")
        checkpoint.CHECKPOINT_NAME_LAST = f"fold{fold_id}_last"
        trainer = pl.Trainer(accelerator="gpu", devices=[gpu], max_epochs=epoch, gradient_clip_val=gradient_clip_val, callbacks=[checkpoint], logger=logger)
        trainer.fit(model, train_loader, valid_loader)
        
        if valid_loader is not None:
            model = LitBertForSequenceClassification.load_from_checkpoint(os.path.join(dirpath, f"fold{fold_id}_{ckpt_name}.ckpt"))
            valid_probs[valid_indices] = torch.cat(trainer.predict(model, valid_loader)).detach().cpu().numpy()
    df["valid_probs"] = valid_probs
    df["valid_preds"] = (valid_probs>=0.5).astype(int)
    df.to_csv(os.path.join(dirpath, "valid_data.csv"))
    logger.log_metrics({"f1":metrics.f1_score(df["state"].values, df["valid_preds"].values)})
    logger.log_metrics({"auroc":metrics.roc_auc_score(df["state"].values, valid_probs)})
    plot_confusion_matrix(df, logger)
        