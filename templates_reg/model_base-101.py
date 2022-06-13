## Pycaret automl codes for regression

import pandas as pd
import pycaret.regression as reg

import os
import pathlib
import time
import datetime
import argparse

import utils as my


## Dataset
dataset = pd.read_csv("./boston-housing.csv",
                      header=None, delimiter=r"\s+",
                      names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                             'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
target_name = "MEDV"
train_size = 0.75


## Modeling Options
model_filename = "base-101"
model_names = ['rf', 'gbr', 'et']                   ## Basic models
# model_names = ['xgboost', 'lightgbm', 'catboost'] ## Advanced models


## Training Options
use_gpu = False
save_model = True
verbose = False


## Feature types: EDA
categorical_features = None
numeric_features = None
date_features = None
ignore_features = None

ordinal_features = None
high_cardinality_features = None

bin_numeric_features = None
group_features = None
group_names = None


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=list, default=model_names)

    ## [1] Data Preparation
    p.add_argument("--imputation_type", type=str, default='simple')
    p.add_argument("--remove_outliers", type=str_to_bool, default=False)
    p.add_argument("--outliers_threshold", type=float, default=0.05)

    ## [2] Scale and Transform
    p.add_argument("--normalize", type=str_to_bool, default=False)
    p.add_argument("--normalize_method", type=str, default='zscore')
    p.add_argument("--transformation", type=str_to_bool, default=False)
    p.add_argument("--transformation_method", type=str, default='yeo-johnson')

    ## [3] Feature Engineering
    p.add_argument("--feature_interaction", type=str_to_bool, default=False)
    p.add_argument("--polynomial_features", type=str_to_bool, default=False)
    p.add_argument("--combine_rare_levels", type=str_to_bool, default=False)
    p.add_argument("--create_clusters", type=str_to_bool, default=False)

    ## [4] Feature Selection
    p.add_argument("--feature_selection", type=str_to_bool, default=False)
    p.add_argument("--feature_selection_threshold", type=float, default=0.8)
    p.add_argument("--remove_multicollinearity", type=str_to_bool, default=False)
    p.add_argument("--multicollinearity_threshold", type=float, default=0.9)
    p.add_argument("--pca", type=str_to_bool, default=False)
    p.add_argument("--pca_method", type=str, default='linear')
    p.add_argument("--pca_components", type=float, default=0.99)
    p.add_argument("--ignore_low_variance", type=str_to_bool, default=False)

    ## [5] Training
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--n_top", type=int, default=3)
    p.add_argument("--metric", type=str, default='R2')

    args = p.parse_args()
    return args


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":

    args = get_args()
    train, test = my.get_data(dataset, train_size, random_state=args.seed)
    start_time = time.time()

    ## Preprocessing
    setup_params = dict(

        ## [1] Data Preparation
        imputation_type = args.imputation_type,
        numeric_features = numeric_features,
        categorical_features = categorical_features,
        date_features = date_features,
        ignore_features = ignore_features,
        ordinal_features = ordinal_features,
        high_cardinality_features = high_cardinality_features,
        remove_outliers = args.remove_outliers,
        outliers_threshold = args.outliers_threshold,

        ## [2] Scale and Transform
        normalize = args.normalize,
        normalize_method = args.normalize_method,
        transformation = args.transformation,
        transformation_method = args.transformation_method,

        ## [3] Feature Engineering
        feature_interaction = args.feature_interaction,
        polynomial_features = args.polynomial_features,
        group_features = group_features,
        group_names = group_names,
        bin_numeric_features = bin_numeric_features,
        combine_rare_levels = args.combine_rare_levels,
        create_clusters = args.create_clusters,

        ## [4] Feature Selection
        feature_selection = args.feature_selection,
        feature_selection_threshold = args.feature_selection_threshold,
        remove_multicollinearity = args.remove_multicollinearity,
        multicollinearity_threshold = args.multicollinearity_threshold,

        pca = args.pca,
        pca_method = args.pca_method,
        pca_components = args.pca_components,
        ignore_low_variance = args.ignore_low_variance,
    )

    session = reg.setup(data=train, target=target_name, session_id=args.seed,
                        silent=True, fold=args.n_folds, verbose=verbose,
                        use_gpu=use_gpu, **setup_params)

    topk = reg.compare_models(n_select=args.n_top, sort=args.metric, verbose=verbose,
                              include=model_names)
    topk_tuned = [reg.tune_model(model, optimize=args.metric, n_iter=args.n_iter,
                        choose_better=True, verbose=verbose) for model in topk]

    bagger = [reg.ensemble_model(model, fold=args.n_folds, optimize=args.metric, method='Bagging',
                                 choose_better=True, verbose=verbose) for model in topk_tuned]
    booster = [reg.ensemble_model(model, fold=args.n_folds, optimize=args.metric, method='Boosting',
                                  choose_better=True, verbose=verbose) for model in topk_tuned]
    blender = reg.blend_models(topk_tuned, fold=args.n_folds, optimize=args.metric,
                            choose_better=True, verbose=verbose)
    stacker = reg.stack_models(topk_tuned, fold=args.n_folds, optimize=args.metric,
                            choose_better=True, verbose=verbose)

    best_automl = reg.automl(optimize=args.metric)
    best_automl = reg.finalize_model(best_automl)

    best_model = reg.get_config('prep_pipe')
    best_model.steps.append(['trained_model', best_automl])
    print(">>", type(best_model.steps[-1][-1]))
    # print(">>", best_model)

    tact = str(datetime.timedelta(seconds=time.time()-start_time)).split(".")[0]


    ## Evaluation
    train_scores = my.get_scores(best_model, train, train[target_name],
                                 name=model_filename, tact=tact)
    test_scores = my.get_scores(best_model, test, test[target_name],
                                name=model_filename, tact=tact)

    result_dir = os.path.join(os.getcwd(), "models")
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
    my.save_results(args, train_scores, test_scores,
                    filename=os.path.join(result_dir, "history.csv"))

    filename = my.get_filename(model_filename, train_scores, test_scores,
                            metrics=['R2', 'RMSE'], random_state=args.seed)
    if save_model:
        reg.save_model(best_model, os.path.join(result_dir, filename))

    print("\n>> Train scores:\n", train_scores)
    print("\n>> Test scores:\n", test_scores, "\n")
    print(">>", filename)
    print(">> %s\n" % tact)

    # best_model_saved = reg.load_model(os.path.join(result_dir, filename))
    # test_scores = my.get_scores(best_model_saved, test, test[target_name],
    #                             name=model_filename)
    # print(">> Test scores (Saved model):\n", test_scores, "\n")
