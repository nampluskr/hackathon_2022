# Hackathon 2022

## Datasets

* Classification
  - [Kaggle] [UCI SECOM Dataset](https://www.kaggle.com/datasets/paresh2047/uci-semcom)
  - [Openml] [UCI-SECOM-Dataset](https://www.openml.org/search?type=data&status=active&id=43587) (id=43587)

```python
secom = fetch_openml(data_id=43587, as_frame=True)
data = pd.concat([secom.data, secom.target], axis=1)
target_name = secom.target_names
```

* Regression
  - [Kaggle] [Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing)
  - [Openml] [Mercedes_Benz_Greener_Manufacturing](https://www.openml.org/search?type=data&status=active&id=42570) (id=42570)

```python
benz = fetch_openml(data_id=42570, as_frame=True)
data = pd.concat([benz.data, benz.target], axis=1)
target_name = benz.target_names
```

* Time Series
  - [Kaggle] [Household Electric Power Consumption](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)


## Pycaret

* Official Documents
  * https://pycaret.gitbook.io/docs/
  * https://pycaret.gitbook.io/docs/get-started/tutorials
  * https://pycaret.gitbook.io/docs/learn-pycaret/examples

* API Reference
  * https://pycaret.readthedocs.io/en/latest/api/classification.html
  * https://pycaret.readthedocs.io/en/latest/api/regression.html
