import subprocess
from itertools import product

def run(files, splits):
    split_values = list(product(*[value for value in splits.values()]))
    split_names = list(splits.keys())
    n_splits = len(split_values)
    subprocess.run('clear') # Linux

    for i, filename in enumerate(files):
        for j, (names, values) in enumerate(zip([split_names]*n_splits, split_values)):
            args_str = ', '.join(["%s=%s" % (k, v) for k, v in zip(names, values)])
            args_run = ([['--' + k, str(v)] for k, v in zip(names, values)])
            args_run = sum(args_run, [])
            args_run = ['python', filename] + args_run

            args_str = ""
            print(">> File[%d/%d] %s" % (i+1, len(files), filename))
            print(">> Split[%d/%d]" % (j+1, len(split_values)), args_str)

            subprocess.run(args_run)


def get_defaults():
    defaults = {}

    ## [1] Data Preparation
    defaults['imputation_type'] = ['simple']
    defaults['fix_imbalance'] = [False]
    defaults['fix_imbalance_method'] = ['smote']
    defaults['remove_outliers'] = [False]
    defaults['outliers_threshold'] = [0.05]

    ## [2] Scale and Transform
    defaults['normalize'] = [True]
    defaults['normalize_method'] = ['robust']
    defaults['transformation'] = [False]
    defaults['transformation_method'] = ['yeo-johnson']

    ## [3] Feature Engineering
    defaults['feature_interaction'] = [False]
    defaults['polynomial_features'] = [False]
    defaults['combine_rare_levels'] = [False]
    defaults['create_clusters'] = [False]

    ## [4] Feature Selection
    defaults['feature_selection'] = [False]
    defaults['feature_selection_threshold'] = [0.8]
    defaults['remove_multicollinearity'] = [False]
    defaults['multicollinearity_threshold'] = [0.9]
    defaults['pca'] = [False]
    defaults['pca_method'] = ['linear']
    defaults['pca_components'] = [0.99]
    defaults['ignore_low_variance'] = [False]

    ## [5] Training
    defaults['seed']    = [111]
    defaults['n_folds'] = [10]
    defaults['n_iter']  = [10]
    defaults['n_top']   = [3]
    defaults['metric']  = ['LogLoss']

    return defaults


if __name__ == "__main__":

    splits = get_defaults()

    ## [1] Data Preparation
    splits['imputation_type'] = ['iterative']              ## 'iterative'
    splits['fix_imbalance'] = [True]
    # splits['fix_imbalance_method'] = ['smote', 'adasyn', 'random']
    # splits['fix_imbalance_method'] = ['smote', 'adasyn', 'random', 'borderline_smote', 'svm_smote']
    splits['remove_outliers'] = [False, True]
    splits['outliers_threshold'] = [0.05]

    ## [2] Scale and Transform
    splits['normalize'] = [False, True]
    splits['normalize_method'] = ['robust', 'simple']             ## 'robust'
    splits['transformation'] = [False]
    # splits['transformation_method'] = ['yeo-johnson']   ## 'quantile'

    ## [3] Feature Engineering
    # splits['feature_interaction'] = [False]
    # splits['polynomial_features'] = [False]
    # splits['combine_rare_levels'] = [False]
    # splits['create_clusters'] = [False]

    ## [4] Feature Selection
    # splits['feature_selection'] = [False]
    # splits['feature_selection_threshold'] = [0.8]
    # splits['remove_multicollinearity'] = [False]
    # splits['multicollinearity_threshold'] = [0.9]
    # splits['pca'] = [False]
    # splits['pca_method'] = ['linear']                   ## 'kernel'
    # splits['pca_components'] = [0.99]                   ## Default 0.99
    # splits['ignore_low_variance'] = [False, True]

    ## [5] Training
    splits['metric']  = ['LogLoss']                         ##, 'AUC', 'Prec.', 'Accuracy']


    files = ['model_base.py']
    run(files, get_defaults())
    run(files, splits)
