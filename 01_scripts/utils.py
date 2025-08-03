from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd

def roc_curve_graph(df, target='bad30', feature='feat', by='X_fold'):
    """
    Compute ROC curve and ROC area for each group (e.g., cross-validation fold).
    """
    fpr = {}
    tpr = {}
    th = {}
    roc_auc = {}
    groups = list(df[by].unique())

    for g in groups:
        fpr[g], tpr[g], th[g] = roc_curve(
            df[df[by] == g][target],
            df[df[by] == g][feature]
        )
        roc_auc[g] = auc(fpr[g], tpr[g])

        # Invert if AUC is below 0.5 (model reversed)
        if roc_auc[g] < 0.5:
            roc_auc[g] = 1 - roc_auc[g]
            tpr[g], fpr[g], th[g] = roc_curve(
                df[df[by] == g][target],
                df[df[by] == g][feature]
            )

    colors = cycle(['cornflowerblue', 'lightblue', 'gray'])
    plt.figure(figsize=(5, 5))
    plt.title("ROC Curve - " + feature)

    for g, color in zip(groups, colors):
        lw = 2
        plt.plot(
            fpr[g], tpr[g], color=color, lw=lw,
            label=str(g) + ' (area = %0.4f)' % roc_auc[g]
        )

    plt.plot([0, 1], [0, 1], color='lightgray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

def model_performance_metrics(model_name, y_test, y_prob, mythreshold):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    AUROC = auc(fpr, tpr)

    y_pred = (y_prob >= mythreshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    Accuracy = (tp + tn) / (tp + tn + fp + fn)
    Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    Recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1_score = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    Total = tn + fp + fn + tp

    show_metrics = pd.DataFrame(data=[[model_name, Accuracy, Precision, Recall, F1_score, AUROC, tn, fp, fn, tp, Total, mythreshold]],
                                columns=['Model_name', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUROC',
                                         'TN', 'FP', 'FN', 'TP', 'Total', 'Threshold'])

    return show_metrics

def mpr_report(model_list, list_of_samp, threshold_list, df, target):
    mpm = pd.DataFrame()

    for mo in model_list:
        for sam in list_of_samp:
            for thr in threshold_list:
                if all(i in [['poso'], ['tmx'], ['mm']] for i in [sam]):
                    # get max score of poso and tmx across all business dt
                    eda = df[df["X_fold"].str.lower().isin(sam)].copy(deep=True)
                    eda = eda.groupby(["cif_no", target])[mo].max().to_frame().reset_index().copy(deep=True)
                else:
                    eda = df[df["X_fold"].isin(sam)].copy(deep=True)

                eda["mypred"] = (eda[mo] >= thr).astype(int)

                tmp = model_performance_metrics(mo, eda[target], eda["mypred"], thr)
                tmp["sample"] = str(sam)

                mpm = pd.concat([mpm, tmp])

    return mpm[["Model_name", "Accuracy", "Precision", "Recall", "F1_score", "AUROC",
                "TN", "FP", "FN", "TP", "Total", "Threshold", "sample"]]

