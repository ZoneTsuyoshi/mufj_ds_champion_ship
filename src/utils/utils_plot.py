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