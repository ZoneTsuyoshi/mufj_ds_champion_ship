import os, json, pathlib, argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import get_test_data, LitBertForSequenceClassification


def test(dirpath, debug=False, ckpt_name=None):
    f = open(os.path.join(dirpath, "config.json"), "r")
    config = json.load(f)
    f.close()
    
    dirpath = dirpath + "/results"
    gpu = config["train"]["gpu"]
    seed = config["train"]["seed"]
    if debug:
        config["train"]["kfolds"] = 2
    kfolds = config["train"]["kfolds"]
    all_weight = config["test"]["all_weight"]
    if ckpt_name is None:
        ckpt_name = config["test"]["ckpt"]
    
    cuda = torch.cuda.is_available()
    print("cuda is avaiable: {}".format(cuda))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    test_loader, test_index = get_test_data(config, debug)
    test_probs = []
    trainer = pl.Trainer(accelerator="gpu", devices=[gpu])
    for i in range(kfolds+1):
        fold_id = i if i<kfolds else "A"
        ckpt_path = os.path.join(dirpath, f"fold{fold_id}_{ckpt_name}.ckpt")
        if os.path.exists(ckpt_path):
            model = LitBertForSequenceClassification.load_from_checkpoint(ckpt_path)
            test_probs.append(torch.cat(trainer.predict(model, test_loader)).detach().cpu().numpy())
    test_probs = np.array(test_probs)
    np.save(os.path.join(dirpath, "test_probs.npy"), test_probs)
    
    test_weights = np.ones(kfolds) if len(test_probs)==kfolds else np.concatenate([np.ones(kfolds), all_weight * np.ones(1)])
    test_probs = (np.array(test_probs) * test_weights[:,None]).sum(0) / (all_weight + kfolds)
    np.save(os.path.join(dirpath, "test_aggregated_probs.npy"), test_probs)
    labels_predicted = (test_probs >= 0.5).astype(int)
    df = pd.DataFrame(labels_predicted.reshape(-1,1))
    df.index = test_index
    df.to_csv(os.path.join(dirpath, "submission.csv"), header=False, index=True)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", action="store_true", help="use best model instead of last model")
    parser.add_argument("-d", action="store_true", help="debug")
    args = parser.parse_args()
    dirpath = os.path.dirname(__file__)
    ckpt_name = "best" if args.b else "last"
    test(dirpath, args.d, ckpt_name)
