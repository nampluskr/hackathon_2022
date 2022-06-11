import pycaret.classification as clf
from sklearn.metrics import log_loss

import os
import pathlib
import time
import datetime

import utils as my


## Modeling Options
model_filename = "default"
model_names = ['rf', 'gbc', 'et']
seed = 111
save_model = True

## Dataset
dataset_path = "./WA_Fn-UseC_-HR-Employee-Attrition.csv"
target_name = "Attrition"
train_size = 0.75


if __name__ == "__main__":

    train, test = my.get_data(dataset_path, train_size, seed, target_name)
    start_time = time.time()

    ## Preprocessing
    session = clf.setup(data=train, target=target_name, session_id=seed, 
                        silent=True, verbose=False)

    ## Training
    clf.remove_metric('MCC')
    clf.remove_metric('Kappa')
    clf.add_metric('logloss', 'LogLoss', log_loss, greater_is_better=False,
                   target="pred_proba")

    topk = clf.compare_models(n_select=3, verbose=False, include=model_names)
    topk_tuned = [clf.tune_model(model, verbose=False, 
                                 choose_better=True) for model in topk]

    blender = clf.blend_models(topk_tuned, choose_better=True, verbose=False)
    stacker = clf.stack_models(topk_tuned, choose_better=True, verbose=False)

    automl = clf.automl()
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
    filename = my.get_filename(model_filename, train_scores, test_scores,
                            metrics=['Accuracy', 'AUC'], random_state=seed)
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