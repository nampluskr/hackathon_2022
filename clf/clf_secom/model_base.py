import pycaret.classification as clf
from sklearn.metrics import log_loss

import os
import pathlib
import time
import datetime
import argparse

import utils as my


## Modeling Options
model_filename = "base"
model_names = ['rf', 'gbc', 'et']
# model_names = ['rf', 'gbc', 'et', 'xgboost', 'lightgbm', 'catboost']


## Training Options
use_gpu = False
save_model = True
verbose = False


## Dataset
dataset_path = "./uci-secom.csv"
target_name = "Pass/Fail"
train_size = 0.75


## Feature types: EDA
categorical_features = None
numeric_features = None
date_features = None
ignore_features = ['Time']
ignore_features += ['509', '506', '256', '313', '234', '399', '186', '401', '375', '141', '191', 
                    '532', '13', '461', '328', '536', '233', '235', '529', '42', '243', '325', 
                    '380', '231', '242', '364', '263', '322', '422', '463', '259', '466', '192', 
                    '69', '327', '450', '374', '538', '262', '237', '49', '507', '398', '178', 
                    '502', '465', '505', '458', '373', '240', '481', '149', '230', '260', '329', 
                    '537', '326', '403', '330', '379', '315', '257', '189', '395', '501', '402', 
                    '179', '462', '372', '531', '464', '530', '370', '535', '369', '115', '533', 
                    '515', '508', '514', '378', '261', '236', '229', '266', '397', '371', '534', 
                    '232', '513', '314', '504', '400', '193', '503', '528', '498', '52', '512', 
                    '264', '265', '284', '451', '226', '258', '241', '394', '381', '97', '414', 
                    '194', '396', '276', '190', '5', '404', '449']
ignore_features += ['360', '74', '206', '307', '197', '140', '384', '267', '137', '321', '156', 
                    '143', '53', '207', '221', '164', '160', '282', '151', '173', '435', '135', 
                    '155', '249', '224', '158', '584', '148', '213', '389', '161', '202', '289', 
                    '244', '138', '579', '565', '183', '291', '358', '443', '288', '187', '251', 
                    '209', '347', '157', '434', '185', '170', '176', '220', '268', '142', '564', 
                    '110', '163', '283', '121', '174', '386', '172', '549', '195', '124', '109', 
                    '154', '198', '246', '175', '159', '583', '144', '253', '212', '554', '147', 
                    '223', '390', '136', '203', '550', '153', '93', '543', '123', '551', '254', 
                    '184', '255', '165', '252', '547', '180', '553', '4', '382', '222', '152', 
                    '383', '166', '169', '205', '574', '181', '312', '219', '572', '245', '66', 
                    '177', '248', '225', '342']

ordinal_features = None
high_cardinality_features = None

bin_numeric_features = None
group_features = None
group_names = None


## Resampling
from imblearn.over_sampling import (SMOTE, ADASYN, RandomOverSampler,
                                    BorderlineSMOTE, SVMSMOTE, KMeansSMOTE)

resampler = {'smote': SMOTE(),
             'adasyn': ADASYN(sampling_strategy='minority'),
             'random': RandomOverSampler(random_state=111),
             'borderline_smote': BorderlineSMOTE(),
             'svm_smote': SVMSMOTE(),
             'kmeans_smote': KMeansSMOTE()}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=list, default=model_names)

    ## [1] Data Preparation
    p.add_argument("--imputation_type", type=str, default='simple')
    p.add_argument("--fix_imbalance", type=str_to_bool, default=False)
    p.add_argument("--fix_imbalance_method", type=str, default='smote')
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
    p.add_argument("--metric", type=str, default='LogLoss')

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
    train, test = my.get_data(dataset_path, train_size, args.seed,
                              target_name=target_name)
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
        fix_imbalance = args.fix_imbalance,
        fix_imbalance_method = (resampler[args.fix_imbalance_method]
                                if args.fix_imbalance else None),
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

    session = clf.setup(data=train, target=target_name, session_id=args.seed, 
                        silent=True, fold=args.n_folds, verbose=verbose, 
                        use_gpu=use_gpu, **setup_params)

    ## Training
    clf.remove_metric('MCC')
    clf.remove_metric('Kappa')
    clf.add_metric('logloss', 'LogLoss', log_loss, greater_is_better=False,
                   target="pred_proba")

    topk = clf.compare_models(n_select=args.n_top, sort=args.metric, verbose=verbose,
                              include=model_names)
    topk_tuned = [clf.tune_model(model, optimize=args.metric, n_iter=args.n_iter,
                            verbose=verbose, choose_better=True) for model in topk]

    blender = clf.blend_models(topk_tuned, fold=args.n_folds, optimize=args.metric,
                            choose_better=True, verbose=verbose)
    stacker = clf.stack_models(topk_tuned, fold=args.n_folds, optimize=args.metric,
                                choose_better=True, verbose=verbose)

    automl = clf.automl(optimize=args.metric)
    automl = clf.finalize_model(automl)

    best = clf.get_config('prep_pipe')
    best.steps.append(['trained_model', automl])
    print(">>", type(best.steps[-1][-1]))

    tact = str(datetime.timedelta(seconds=time.time()-start_time)).split(".")[0]


    ## Evaluation
    train_scores = my.get_scores(best, train, train[target_name], 
                                 name=model_filename, tact=tact)
    test_scores = my.get_scores(best, test, test[target_name], 
                                name=model_filename, tact=tact)

    result_dir = os.path.join(os.getcwd(), "models")
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
    my.save_results(args, train_scores, test_scores,
                    filename=os.path.join(result_dir, "history.csv"))

    filename = my.get_filename(model_filename, train_scores, test_scores,
                            metrics=['Accuracy', 'AUC'], random_state=args.seed)
    if save_model:
        clf.save_model(best, os.path.join(result_dir, filename))

    print("\n>> Train scores:\n", train_scores)
    print("\n>> Test scores:\n", test_scores, "\n")
    print(">>", filename)
    print(">> %s\n" % tact)

    # best_saved = clf.load_model(filename)
    # test_scores = my.get_scores(best_saved, test, test[target_name],
    #                             name=model_filename)
    # print(">> Test scores (Saved model):\n", test_scores, "\n")