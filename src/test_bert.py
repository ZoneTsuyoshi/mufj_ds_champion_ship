import os, json, pathlib, argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import get_train_data, get_test_data, LitBertForSequenceClassification, threshold_search


def test(dirpath, debug=False, pseudo_labeling_vars=None, with_valid_on=False, ckpt_name=None, all_weight=None):
    f = open(os.path.join(dirpath, "config.json"), "r")
    config = json.load(f)
    f.close()
    
    dirpath = dirpath + "/results"
    gpu = config["train"]["gpu"]
    seed = config["train"]["seed"]
    if debug:
        config["train"]["kfolds"] = 2
    kfolds = config["train"]["kfolds"]
    if ckpt_name is None:
        ckpt_name = config["test"]["ckpt"]
    if all_weight is None:
        all_weight = config["test"]["all_weight"]
    
    cuda = torch.cuda.is_available()
    print("cuda is avaiable: {}".format(cuda))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    if not os.path.exists(os.path.join(dirpath, "test_probs.npy")):
        if with_valid_on:
            _, valid_loader_list, _, valid_indices_list, _, _, df = get_train_data(config, debug, pseudo_labeling_vars)
            valid_probs = np.zeros(len(df))
        test_loader, test_index = get_test_data(config, debug)
        test_probs = []
        trainer = pl.Trainer(accelerator="gpu", devices=[gpu])
        for i in range(kfolds+1):
            fold_id = i if i<kfolds else "A"
            ckpt_path = os.path.join(dirpath, f"fold{fold_id}_{ckpt_name}.ckpt")
            if os.path.exists(ckpt_path):
                model = LitBertForSequenceClassification.load_from_checkpoint(ckpt_path)
                test_probs.append(torch.cat(trainer.predict(model, test_loader)).detach().cpu().numpy())
                if with_valid_on and valid_loader_list[i] is not None:
                    valid_probs[valid_indices_list[i]] = torch.cat(trainer.predict(model, valid_loader_list[i])).detach().cpu().numpy()
        test_probs = np.array(test_probs)
        np.save(os.path.join(dirpath, "test_probs.npy"), test_probs)
    else:
        test_probs = np.load(os.path.join(dirpath, "test_probs.npy"))
    
    if with_valid_on:
        df["valid_probs"] = valid_probs
        df["valid_preds"] = (valid_probs>=0.5).astype(int)
        df.to_csv(os.path.join(dirpath, "valid_data.csv"))
        search_results = threshold_search(df["state"].values, df["valid_probs"].values)
        # logger.log_metrics({"f1-half":metrics.f1_score(df["state"].values, df["valid_preds"].values),
        #                     "f1":search_results["f1"], "threshold":search_results["threshold"],
        #                     "auroc":metrics.roc_auc_score(df["state"].values, valid_probs)})
        with open(os.path.join(dirpath, "search_results.json"), "w") as f:
            json.dump(search_results, f, indent=4, ensure_ascii=False)
            
    f = open(os.path.join(dirpath, "search_results.json"), "r")
    search_results = json.load(f)
    f.close()
    
    test_weights = np.ones(kfolds) if len(test_probs)==kfolds else np.concatenate([np.ones(kfolds), all_weight * np.ones(1)])
    test_probs = (np.array(test_probs) * test_weights[:,None]).sum(0) / (all_weight + kfolds)
    np.save(os.path.join(dirpath, "test_aggregated_probs.npy"), test_probs)
    labels_predicted = (test_probs >= search_results["threshold"]).astype(int)
    df = pd.DataFrame(labels_predicted.reshape(-1,1))
    df.index = test_index
    df.to_csv(os.path.join(dirpath, "submission.csv"), header=False, index=True)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bert', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", action="store_true", help="use the best model")
    parser.add_argument("-l", action="store_true", help="use the last model")
    parser.add_argument("-d", action="store_true", help="debug")
    parser.add_argument("-p", default=None, nargs="*", help="pseudo labeling, 1st: exp_id, 2nd: confidence")
    parser.add_argument("-v", action="store_true", help="with valid")
    parser.add_argument("-w", "--all_weight", default=None, type=float, help="weight of all training")
    args = parser.parse_args()
    dirpath = os.path.dirname(__file__)
    ckpt_name = None
    if args.b: ckpt_name = "best"
    if args.l: ckpt_name = "last"
    test(dirpath, args.d, args.p, args.v, ckpt_name)
