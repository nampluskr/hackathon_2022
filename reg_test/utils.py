import os
import pandas as pd
from pycaret.utils import check_metric
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_data(dataset_path, train_size, random_state):
    dataset = pd.read_csv(dataset_path)

    train = dataset.sample(frac=train_size, random_state=random_state)
    test = dataset.drop(train.index)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    
    print('Train Data:', train.shape)
    print('Test  Data:', test.shape, '\n')
    return train, test


def get_scores(model, data, target, name=None, tact=None):
    pred = model.predict(data)

    scores = {}
    scores['Model'] = [name] if name is not None else ['']
    scores['MAE'] = [mean_absolute_error(target, pred)]
    scores['MSE'] = [mean_squared_error(target, pred)]
    scores['RMSE'] = [mean_squared_error(target, pred, squared=False)]
    scores['R2'] = [r2_score(target, pred)]
    scores['RMSLE'] = [check_metric(target, pred, metric='RMSLE')]
    scores['MAPE'] = [check_metric(target, pred, metric='MAPE')]
    scores['Tact'] = [tact] if tact is not None else ['']

    df = pd.DataFrame(scores)
    df.iloc[:, 1:-1] = df.iloc[:, 1:-1].apply(lambda x: round(x, 4))
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