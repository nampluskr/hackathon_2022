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

            if file['verbose']: args_run += ['--verbose']
            if file['save']:    args_run += ['--save']
            if file['use_gpu']: args_run += ['--use_gpu']

            print(">> File[%d/%d] %s" % (i+1, len(files), file['filename']))
            print(">> Split[%d/%d]" % (j+1, len(args_values)), args_str)
            subprocess.run(args_run)


if __name__ == "__main__":

    args = {}
    args['seed']    = [123]                       ## Default: 123
    # args['n_folds'] = [5]                       ## Default: 10
    # args['n_iter']  = [10, 20, 50]              ## Default: 10
    # args['n_top']   = [3]                       ## Default: 3
    # args['metric']  = ['AUC', 'LogLoss']        ## Dafault: LogLoss
    
    file1 = dict(filename='model_base-001.py', save=True, verbose=False, use_gpu=False)
    file2 = dict(filename='model_base-002.py', save=True, verbose=False, use_gpu=False) 
    
    option = [0, 1, 0]

    if option[0]: run([file1], args)
    if option[1]: run([file2], args)
    if option[2]: run([file2, file1], args)
