from pickletools import optimize
import pycaret.classification as clf
from sklearn.metrics import log_loss

import time
import datetime
import argparse

import utils as my


## https://www.kaggle.com/datasets/paresh2047/uci-semcom
dataset_path = "../../hackathon/data/uci-secom.csv"
target_name = "Pass/Fail"


## Parameters for models
model_filename = "base-101"
model_names = ['rf', 'gbc', 'et']                    ## 'xgboost', 'lightgbm', 'catboost'

model_params = {name: {} for name in model_names}    ## model_params['rf'] = dict(max_depth=5)
model_grids = {name: None for name in model_names}   ## model_grids['rf'] = dict(max_depth=[5, 10])


## Parameters for preprocessing
pre_params = dict(
    ignore_features=['Time'],
    categorical_features=None,
    ordinal_features=None,
    numeric_features=None,
    date_features=None,

    bin_numeric_features=None,
    high_cardinality_features=None,
    group_features=None,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_size", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--use_gpu", action='store_const', const=True, default=False)
    p.add_argument("--save", action='store_const', const=True, default=False)
    p.add_argument("--verbose", action='store_const', const=True, default=False)

    p.add_argument("--models", type=list, default=model_names)
    p.add_argument("--metric", type=str, default='LogLoss')
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--n_top", type=int, default=3)

    args = p.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    start_time = time.time()

    ## Preprocessing
    train, test = my.get_data(dataset_path, args.train_size, args.seed, target_name=target_name)
    session = clf.setup(data=train, target=target_name, session_id=args.seed, silent=True,
              fold=args.n_folds, verbose=args.verbose, use_gpu=args.use_gpu, **pre_params)


    ## Training
    clf.remove_metric('MCC')
    clf.remove_metric('Kappa')
    clf.add_metric('logloss', 'LogLoss', log_loss, greater_is_better=False, target="pred_proba")

    topk = {name: clf.create_model(name, fold=args.n_folds, verbose=args.verbose, 
                                   **model_params[name]) for name in model_names}
    topk_tuned = {name: clf.tune_model(topk[name], optimize=args.metric, n_iter=args.n_iter,
                        verbose=args.verbose, custom_grid=model_grids[name]) for name in model_names}
    
    blender = clf.blend_models(list(topk_tuned.values()), fold=args.n_folds, 
                               choose_better=True, verbose=args.verbose)
    # stacker = clf.stack_models(topk_tuned, fold=args.n_folds, choose_better=True, verbose=args.verbose)

    automl = clf.automl(optimize=args.metric)
    automl = clf.finalize_model(automl)

    best = clf.get_config('prep_pipe')
    best.steps.append(['trained_model', automl])

    tact = str(datetime.timedelta(seconds=time.time()-start_time)).split(".")[0]

    ## Evaluation
    train_scores = my.get_scores(best, train, train[target_name], name=model_filename, tact=tact)
    test_scores = my.get_scores(best, test, test[target_name], name=model_filename, tact=tact)
    my.save_results(args, train_scores, test_scores, filename="history.csv")

    filename = my.get_filename(model_filename, train_scores, test_scores, 
                               metrics=['Accuracy', 'AUC'], random_state=args.seed)
    if args.save:
        clf.save_model(best, filename)

    print("\n>> Train scores:\n", train_scores)
    print("\n>> Test scores:\n", test_scores, "\n")
    print(">>", filename)
    print(">> %s\n" % tact)

    # best_saved = clf.load_model(filename)
    # test_scores = my.get_scores(best_saved, test, test[target_name], name=model_filename)
    # print(">> Test scores (Saved model):\n", test_scores, "\n")