from pickletools import optimize
import pycaret.classification as clf
from sklearn.metrics import log_loss

import time
import datetime
import argparse

import utils as my


## Modeling Options
model_filename = "base-002"
model_names = ['rf', 'gbc', 'et']
# model_names = ['rf', 'gbc', 'et', 'xgboost', 'lightgbm', 'catboost']


## Training Options
use_gpu = False
save_model = True
verbose = False


## Dataset Infomation
dataset_path = "../../hackathon/data/uci-secom.csv"
target_name = "Pass/Fail"
train_size = 0.75

ignore_features=['Time']
categorical_features=None
ordinal_features=None
numeric_features=None
date_features=None
# bin_numeric_features=None,
# high_cardinality_features=None,
# group_features=None,


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--models", type=list, default=model_names)
    p.add_argument("--metric", type=str, default='LogLoss')

    ## preprocessing
    p.add_argument("--pca", type=bool, default=False)
    p.add_argument("--pca_components", type=float, default=0.99)

    p.add_argument("--normalize", type=bool, default=False)
    p.add_argument("--normalize_method", type=str, default='zscore')

    p.add_argument("--remove_outliers", type=bool, default=False)
    p.add_argument("--outliers_threshold", type=float, default=0.05)

    p.add_argument("--fix_imbalance", type=bool, default=False)

    ## training options
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--n_top", type=int, default=3)

    args = p.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    train, test = my.get_data(dataset_path, train_size, args.seed, target_name=target_name)

    start_time = time.time()

    ## Preprocessing
    setup_params = dict(

        ## Feature types:
        ignore_features=ignore_features,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        numeric_features=numeric_features,
        date_features=date_features,
        # bin_numeric_features=bin_numeric_features,
        # high_cardinality_features=high_cardinality_features,
        # group_features=group_features,

        fix_imbalance=args.fix_imbalance,

        normalize=args.normalize,
        normalize_method=args.normalize_method,

        pca=args.pca,
        pca_components=args.pca_components,

        remove_outliers=args.remove_outliers,
        outliers_threshold=args.outliers_threshold,

        # remove_multicollinearity=True,
        # multicollinearity_threshold=0.95,  ## Default: 0.9
        # transformation=True,
        # ignore_low_variance=True,
    )

    session = clf.setup(data=train, target=target_name, session_id=args.seed, silent=True,
              fold=args.n_folds, verbose=verbose, use_gpu=use_gpu, **setup_params)


    ## Training
    clf.remove_metric('MCC')
    clf.remove_metric('Kappa')
    clf.add_metric('logloss', 'LogLoss', log_loss, greater_is_better=False, target="pred_proba")

    topk = clf.compare_models(n_select=args.n_top, sort=args.metric, verbose=verbose, include=model_names)
    topk_tuned = [clf.tune_model(model, optimize=args.metric, n_iter=args.n_iter, verbose=verbose,
                                 choose_better=True) for model in topk]

    blender = clf.blend_models(topk_tuned, fold=args.n_folds, optimize=args.metric,
                               choose_better=True, verbose=verbose)
    # stacker = clf.stack_models(topk_tuned, fold=args.n_folds, optimize=args.metric,
    #                            choose_better=True, verbose=verbose)

    automl = clf.automl(optimize=args.metric)
    automl = clf.finalize_model(automl)

    best = clf.get_config('prep_pipe')
    best.steps.append(['trained_model', automl])
    print(">>", type(best.steps[-1][-1]))

    tact = str(datetime.timedelta(seconds=time.time()-start_time)).split(".")[0]


    ## Evaluation
    train_scores = my.get_scores(best, train, train[target_name], name=model_filename, tact=tact)
    test_scores = my.get_scores(best, test, test[target_name], name=model_filename, tact=tact)
    my.save_results(args, train_scores, test_scores, filename="history.csv")

    filename = my.get_filename(model_filename, train_scores, test_scores,
                               metrics=['Accuracy', 'AUC'], random_state=args.seed)
    if save_model:
        clf.save_model(best, filename)

    print("\n>> Train scores:\n", train_scores)
    print("\n>> Test scores:\n", test_scores, "\n")
    print(">>", filename)
    print(">> %s\n" % tact)

    # best_saved = clf.load_model(filename)
    # test_scores = my.get_scores(best_saved, test, test[target_name], name=model_filename)
    # print(">> Test scores (Saved model):\n", test_scores, "\n")