## User-defined functions for classification

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.stats import pearsonr, chi2_contingency

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, log_loss
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_data(dataset, train_size, random_state, target_name=None):
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


def get_nunique_features(dataset, value, kind="equal"):
    features = dataset.nunique().sort_values()
    if kind == 'greater_than':
        features = features[features.values > value].index
    elif kind == 'less_than':
        features = features[features.values < value].index
    else:
        features = features[features.values == value].index
    return features.tolist()


def show_counts(dataset, feature_names, color='green'):
    features = dataset[feature_names].nunique().sort_values()
    n_features = len(feature_names)
    x, y = features.values, range(n_features)

    fig, ax = plt.subplots(figsize=(15, n_features/2.5 + 0.4))
    ax.barh(y, features.values, align='center', color=color)
    ax.set_yticks(y, labels=features.index)
    ax.set_xlabel("Counts", fontsize=12)
    ax.tick_params(labelsize=12)
    for xi, yi in zip(x, y):
        ax.text(xi, yi, xi, fontdict=dict(size=12, color='red'))

    fig.tight_layout()
    plt.show()


def show_histogram(dataset, feature_names, target=None, n_cols=6, width=15, kde=False, xlabels=True):
    n_features = len(feature_names)
    n_rows = n_features // n_cols + (1 if n_features % n_cols else 0)
    height = width*n_rows/n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height))
    for i, ax in enumerate(axes.flat):
        if i < n_features:
            sns.histplot(dataset, x=feature_names[i], hue=target,
                         stat="percent", multiple="dodge", kde=kde, ax=ax)
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
            ax.set_ylabel('')
            if not xlabels:
                ax.xaxis.set_ticklabels([])

        else:
            ax.set_axis_off()
    fig.tight_layout()
    plt.show()


## Correlation coefficients for numerical features
def get_high_corr_numerical(dataset, feature_names, threshold=0.9):
    corr_names = np.array(list(combinations(feature_names, 2)))
    corr_values = np.array([])
    for name1, name2 in corr_names:
        corr = pearsonr(dataset[name1], dataset[name2])[0]
        corr_values = np.append(corr_values, corr)

    indices = corr_values.argsort()
    names = corr_names[indices][::-1]
    values = corr_values[indices][::-1]

    high_corr_names = names[np.abs(values) >= threshold]
    high_corr_values = values[np.abs(values) >= threshold]

    names = np.array([' vs. '.join(name) for name in names])
    for name, value in zip(high_corr_names, high_corr_values):
        print("Pearson's R = %.4f - %s" % (value, name))

    return np.unique(high_corr_names[:, 0]).tolist(), (names, values)


## Correlation coefficients for categorical features
def cramerv(var1, var2) :
    crosstab =np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    stat = chi2_contingency(crosstab)[0]    # Chi2 test
    obs = crosstab.sum()                  # Number of observations
    mini = min(crosstab.shape) - 1          # Minimum value between the columns and the rows of the cross table
    return stat/(obs*mini)


def get_high_corr_categorical(dataset_encoded, feature_names, threshold=0.9):
    corr_names = np.array(list(combinations(feature_names, 2)))
    corr_values = np.array([])
    for name1, name2 in corr_names:
        corr = cramerv(dataset_encoded[name1], dataset_encoded[name2])
        corr_values = np.append(corr_values, corr)

    indices = corr_values.argsort()
    names = corr_names[indices][::-1]
    values = corr_values[indices][::-1]

    high_corr_names = names[np.abs(values) >= threshold]
    high_corr_values = values[np.abs(values) >= threshold]

    names = np.array([' vs. '.join(name) for name in names])
    for name, value in zip(high_corr_names, high_corr_values):
        print("Cramer's V = %.4f - %s" % (value, name))

    return np.unique(high_corr_names[:, 0]).tolist(), (names, values)


def show_correlations(names, values, threshold=0.5):
    corr_names = names[abs(values) > threshold]
    corr_values = values[abs(values) > threshold]
    
    fig, ax = plt.subplots(figsize=(15, len(corr_values)/2))
    ax.barh(corr_names, corr_values, align='center')
    ax.tick_params(labelsize=12)
    ax.set_xlabel
    for xi, yi in zip(corr_values, range(len(corr_values))):
        ax.text(xi, yi, '%.4f' % xi, fontdict=dict(size=12, color='red'))
    ax.invert_yaxis()
    ax.grid()
    fig.tight_layout()
    plt.show()


def show_history(df_history, mode, sort_by=['Accuracy']):
    def change_color(value):
        if isinstance(value, bool) and value == True:
            color = 'red'
        elif isinstance(value, bool) and value == False:
            color = 'gray'
        elif isinstance(value, str):
            color = 'blue'
        elif isinstance(value, float):
            color = 'black'
        else:
            color = 'gray'
        return 'color: %s' % color

    history = df_history[df_history["Mode"] == mode]
    history = history.sort_values(by=sort_by, ascending=False).T
    if mode == "Train":
        history.columns += 1
    history = history.style.applymap(change_color)
    return history