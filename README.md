# Hackathon 2022

## Datasets

* Classification
  - [Kaggle] [UCI SECOM Dataset](https://www.kaggle.com/datasets/paresh2047/uci-semcom)
  - [Kaggle] [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
  - [Kaggle] [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
  - [Kaggle] [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

* Regression
  - [Kaggle] [Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)
  - [Kaggle] [The Boston Housing Dataset](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset/notebook)

* Time Series
  - [Kaggle] [Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)

## Preprocessing

### Selected Methods
![preprocessing](./Preprocessing.jpg)

## Basic Codes

### Classification

```python
import pycaret.classification as clf

session = clf.setup(data=train, target=target_name)

```

### Regression

```python
from pycaret.datasets import get_data
import pycaret.regression as reg

boston = get_data('boston')
session = reg.setup(data=boston, target = 'medv', silent=True, verbose=False)

topk = reg.compare_models(n_select=3, include=['rf', 'gbr', 'et'])
topk_tuned = [reg.tune_model(model) for model in topk]

blender = reg.blend_models(topk_tuned)
stacker = reg.stack_models(topk_tuned)

best_automl = reg.automl(optimize='MAE')
best_automl = reg.finalize_model(best_automl)

best_model = reg.get_config('prep_pipe')
best_model.steps.append(['trained_model', best_automl])
print(">>", type(best_model.steps[-1][-1]))
```


## References

* Official Documents
  - https://pycaret.gitbook.io/docs/
  - https://pycaret.gitbook.io/docs/get-started/tutorials
  - https://pycaret.gitbook.io/docs/learn-pycaret/examples

* API Reference
  - https://pycaret.readthedocs.io/en/latest/api/classification.html
  - https://pycaret.readthedocs.io/en/latest/api/regression.html
