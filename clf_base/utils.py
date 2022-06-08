import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, log_loss
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_data(dataset_path, train_size, random_state, target_name=None):
    dataset = pd.read_csv(dataset_path)

    if target_name is None:
        train = dataset.sample(frac=train_size, random_state=random_state)
    else:
        target = dataset[target_name]
        dataset[target_name] = LabelEncoder().fit_transform(target)
        train = dataset.groupby(target_name, group_keys=False).apply(
            lambda x: x.sample(frac=train_size, random_state=random_state))

    test = dataset.drop(train.index)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    
    print('Train Data:', train.shape)
    print('Test  Data:', test.shape, '\n')
    return train, test


def get_scores(model, data, target, name=None, tact=None):
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)

    scores = {}
    scores['Model'] = [name] if name is not None else ['']
    scores['Accuracy'] = [accuracy_score(target, pred)]
    scores['AUC'] = [roc_auc_score(target, pred_proba[:, 1])]
    scores['Recall'] = [recall_score(target, pred)]
    scores['Prec.'] = [precision_score(target, pred)]
    scores['F1'] = [f1_score(target, pred)]
    # scores['MCC'] = [matthews_corrcoef(target, pred)]
    # scores['Kappa'] = [cohen_kappa_score(target, pred)]
    scores['LogLoss'] = [log_loss(target, pred_proba[:, 1])]

    df = pd.DataFrame(scores)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: round(x, 4))
    df['Tact'] = [tact] if tact is not None else ['']
    return df


def get_filename(model_name, train_scores, test_scores, metrics, random_state):
    metric1, metric2 = metrics

    train_score1 = train_scores[metric1].values[0]
    train_score2 = train_scores[metric2].values[0]
    test_score1 = test_scores[metric1].values[0]
    test_score2 = test_scores[metric2].values[0]

    seed = "seed-%d" % random_state
    train_score = "train__%s-%.4f_%s-%.4f" % (metric1, train_score1, metric2, train_score2)
    test_score = "test__%s-%.4f_%s-%.4f" % (metric1, test_score1, metric2, test_score2)
    return '__'.join([model_name, train_score, test_score, seed])


def save_results(args, train_scores, test_scores, filename):
    df_args = pd.DataFrame(vars(args)).drop(["models"], axis=1).drop_duplicates()
    df_args["models"] = ', '.join(args.models)
    df_train = pd.concat([train_scores, df_args], axis=1)
    df_test = pd.concat([test_scores, df_args], axis=1)

    results = pd.concat([df_train, df_test], ignore_index=True)
    results['Mode'] = ['Train', 'Test']
    results.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
    return results


def show_confusion_matrix(model, data, target, labels=None):
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    pred = model.predict(data)
    cm = confusion_matrix(target, pred, labels=labels)
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=labels)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm/cm.sum(),
                                display_labels=labels)
    disp1.plot(cmap='viridis', ax=ax1)
    disp2.plot(cmap='viridis', ax=ax2)

    for ax in (ax1, ax2):
        ax.grid(False)
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Predicted label", fontsize=15)
        ax.set_ylabel("True label", fontsize=15)

    fig.tight_layout()
    plt.show()


def show_roc_curve(model, data, target):
    fig, ax = plt.subplots(figsize=(6, 4))

    proba = model.predict_proba(data)
    fpr, tpr, _ = roc_curve(target, proba[:, 1])
    ax.plot(fpr, tpr, 'b', lw=2, label='ROC curve (AUC = %.4f)' % auc(fpr, tpr))
    ax.plot([0, 1], [0, 1], 'k:', lw=1, linestyle="--")

    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.legend(loc="lower right", fontsize=12)

    fig.tight_layout()
    plt.show()