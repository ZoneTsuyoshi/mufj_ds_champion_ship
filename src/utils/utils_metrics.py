from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(df, logger):
    confmat = metrics.confusion_matrix(df["state"].values, df["valid_preds"].values)
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(confmat / confmat.sum(1)[:,None], cmap="Blues", ax=ax, vmin=0, vmax=1, square=True, annot=True, fmt=".2f")
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    logger.experiment.log_figure("confusion matrix", fig)
    
    
def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = metrics.f1_score(y_true, y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result