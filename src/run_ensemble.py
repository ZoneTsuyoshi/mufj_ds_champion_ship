import os, datetime, json, argparse
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize
from utils import threshold_search


def get_loss_fn(probs, valid_labels, loss_name="f1", th_searh=False):
    if loss_name in ["CE", "CEL"]:
        def loss_fn(weight):
            final_probs = (probs * weight[:,None]).sum(0)
            return metrics.log_loss(valid_labels, final_probs)
    elif loss_name in ["F1", "f1"]:
        def loss_fn(weight):
            # weight = weight / weight.sum()
            final_probs = (probs * weight[:,None]).sum(0)
            if th_searh:
                return -threshold_search(valid_labels, final_probs)["f1"]
            else:
                final_preds = (final_probs >= 0.5).astype(int)
                return -metrics.f1_score(valid_labels, final_preds)
    return loss_fn


def main(dirpath, ensemble_names, method="Nelder-Mead", loss_name="f1", th_searh=False, debug=False):
    valid_probs_list, test_probs_list = [], []
    for name in ensemble_names:
        valid_probs_list.append(pd.read_csv(os.path.join("exp", name, "results/valid_data.csv"), index_col=0)["valid_probs"].values)
        test_probs_list.append(np.load(os.path.join("exp", name, "results/test_aggregated_probs.npy")))
    valid_probs = np.array(valid_probs_list)
    valid_labels = pd.read_csv("data/train.csv", index_col=0)["state"].values
    test_probs = np.array(test_probs_list)
    loss_fn = get_loss_fn(valid_probs, valid_labels, loss_name, th_searh)
    
    if method=="random":
        np.random.seed(123)
        n_cands = 1000
        cand_w = np.random.rand(n_cands, len(ensemble_names))
        cand_w = cand_w / cand_w.sum(1)[:,None]
        scores = np.zeros(n_cands)
        for i,w in enumerate(cand_w):
            scores[i] = loss_fn(w)
        weight = cand_w[np.argmin(scores)]
    else:
        initial_w = np.ones(len(ensemble_names), dtype=float) / len(ensemble_names)
        cons = ({'type':'eq','fun':lambda w: 1-w.sum()})
        bounds = [(0,1)]*len(ensemble_names)
        results = minimize(loss_fn, initial_w, method=method, bounds=bounds, constraints=cons)
        weight = results["x"]
    search_results = threshold_search(valid_labels, (valid_probs * weight[:,None]).sum(0))
    print("best f1: {:.3f}".format(search_results["f1"]))
    result_dict = {"models":ensemble_names, "target":loss_name, "method":method, "weight":weight.tolist(), "f1":search_results["f1"], "threshold":search_results["threshold"]}
    with open(os.path.join(dirpath, "results.json"), "w") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)
    
    labels_predicted = ((test_probs * weight[:,None]).sum(0) >= search_results["threshold"]).astype(int)
    test_index = pd.read_csv("data/test.csv", index_col=0).index
    df = pd.DataFrame(labels_predicted.reshape(-1,1))
    df.index = test_index
    df.to_csv(os.path.join(dirpath, "submission.csv"), header=False, index=True)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ensemble', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", action="store_true", help="debug")
    parser.add_argument("-t", action="store_true", help="threshold search")
    parser.add_argument("-m", type=str, default="Nelder-Mead", help="method")
    parser.add_argument("-l", type=str, default="f1", help="loss")
    parser.add_argument("-e", default=None, nargs="*", help="ensemble names")
    args = parser.parse_args()
    dirpath = os.path.dirname(__file__)
    main(dirpath, args.e, args.m, args.l, args.t, args.d)