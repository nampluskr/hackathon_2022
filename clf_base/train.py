import subprocess
from itertools import product

def run(files, args):
    args_values = list(product(*[args[key] for key in args]))
    args_keys = [[key for key in args]]*len(args_values)
    subprocess.run('clear')

    for i, file in enumerate(files):
        for j, (keys, values) in enumerate(zip(args_keys, args_values)):
            args_str = ', '.join(["%s=%s" % (k, v) for k, v in zip(keys, values)])
            args_run = ([['--' + k, str(v)] for k, v in zip(keys, values)])
            args_run = sum(args_run, [])
            args_run = ['python', file['filename']] + args_run

            print(">> File[%d/%d] %s" % (i+1, len(files), file['filename']))
            print(">> Split[%d/%d]" % (j+1, len(args_values)), args_str)
            subprocess.run(args_run)


if __name__ == "__main__":

    T, F = 'True', 'False'

    args = {}
    args['seed']    = [111]                       ## Default: 123
    # args['n_folds'] = [5]                       ## Default: 10
    # args['n_iter']  = [10, 20, 50]              ## Default: 10
    # args['n_top']   = [3]                       ## Default: 3
    # args['metric']  = ['AUC', 'LogLoss']        ## Dafault: LogLoss
    
    args['pca'] = [T]
    args['pca_components'] = [0.95]   ## Default 0.99
    
    args['normalize'] = [T]
    args['normalize_method'] = ['zscore', 'robust']
    
    args['remove_outliers'] = [T]
    args['outliers_threshold'] = [0.05]
    
    args['fix_imbalance'] = [T]
    
    file1 = dict(filename='model_base-002.py')
    file2 = dict(filename='model_base-002.py')
    
    option = 1

    if option == 1: run([file1], args)
    if option == 2: run([file2], args)
    if option == 3: run([file1, file2], args)