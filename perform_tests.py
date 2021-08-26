def staged_scores(model, X, y):
  '''helper function'''
  if isinstance(model, (AdaBoostClassifier, AdaBoostRegressor)):
    return np.array(list(model.staged_score(X, y)))
  if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier,
                          BaggingClassifier)):
    probas = np.array([e.predict_proba(X) for e in model.estimators_])
    cumulative_probas = np.cumsum(probas, axis=0)
    cumulative_preds = cumulative_probas.argmax(axis=-1)
    scores = np.array([accuracy_score(y, p) for p in cumulative_preds])
    return scores
  elif isinstance(model, (RandomForestRegressor, ExtraTreesRegressor,
                          BaggingRegressor)):
    preds = np.array([e.predict(X) for e in model.estimators_])
    cumulative_preds = np.cumsum(preds, axis=0)
    cumulative_preds /= (1 + np.arange(len(preds)))[:, None]
    scores = np.array([r2_score(y, p) for p in cumulative_preds])
    return scores

# Example:
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# X, y = fetch_openml(name='arrhythmia', return_X_y=True, version=2)
# X = np.nan_to_num(X)
# y = (y == 'N').astype(int) #'N'->0, 'P'->1
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# model = BaggingClassifier(n_estimators=100).fit(X_train, y_train)
# staged_scores(model, X_train, y_train)

#!pip install imbalanced-learn -q

import itertools
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm as _tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
        ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier, \
        AdaBoostRegressor, BaggingClassifier, BaggingRegressor
from sklearn.metrics import accuracy_score, r2_score
import sklearn.base
import time

def perform_tests(base_model, datasets, folds=10, hyperparams=None, oversample_test=False, return_models=False, tqdm=True, staged=True):
  '''
  Fitting and evaluating model by StratifiedKFold cross-validation,
  possibly using multiple datasets and/or multiple hyperparameter values.

  Parameters:
  ----------
  base_model :
    Sklearn estimator, for example LogisticRegression().
  dataset:
    Tuple (X, y) or array of (X, y) or dict: name->(X, y).
  folds:
    Folds count, >=1.
  hyperparams:
    Optional: dict: name->(values list). If parameter starts with ':',
    it is set for base_estimator, useful for sklearn.ensemble models.
    Example: hyperparams={'n_estimators': range(1, 100), ':max_depth': [3, 7]}
  oversample_test:
    Should oversample to make test folds balanced?
  return_models:
    Should also return all trained models?
  tqdm:
    Should show progress by tqdm widget? Pass 2 to also use tqdm for folds
  staged:
    Used if 'n_estimators' hyperparameter is specified in hyperparams dict.
    If staged==True, will use staged predictions on same model instead of
    fitting new model for each n_estimators value. This can reduce the calculation
    time by many times and should not affect result. This optimization works
    only with RandomForest, ExtraTrees, AdaBoost and Bagging ensembles from sklearn.

  Returns:
  ----------
  If hyperparams is None:
    Numpy array of shape (datasets_count, folds_count, 2), where
    all axes with length 1 are removed. Last axis means 0=train, 1=test.
  If dataset was a dict, then using list of it's .values().
    (in python 3.7 dictionary order is guaranteed to be insertion order)
  If N hyperparams are specified:
    Same array, but prepending N axes for N hyperparams.
    Hyperparams go in the same order as in 'hyperparams' param.
    Axes corresponding to hyperparams with single value are not removed.
  If return_models:
    Also returns numpy array of all trained models. This array has
    the same shape, but without last axis (train/test).
  '''
  if oversample_test:
    from imblearn.over_sampling import RandomOverSampler
  # tqdm
  def maybe_tqdm(iterable, title=None):
    arr = list(iterable)
    if tqdm and len(arr) > 1 and (title != 'fold' or tqdm >= 2):
      return _tqdm(arr, desc=title, leave=False)
    return arr
  # preparing datasets
  if type(datasets) == tuple or (type(datasets) != dict \
                                 and isinstance(datasets[0], np.ndarray)):
    datasets = [datasets]
  elif type(datasets) == dict:
    datasets = list(datasets.values())
  # hyperparams
  hyperparams_were_none = (hyperparams is None)
  if hyperparams is None:
    hyperparams = {'nothing': [0]}
  # n_estimators performance tweak
  use_staged_scores = staged and 'n_estimators' in hyperparams and not return_models \
         and isinstance(base_model, (RandomForestClassifier, RandomForestRegressor,
         ExtraTreesClassifier, ExtraTreesRegressor, AdaBoostClassifier,
         AdaBoostRegressor, BaggingClassifier, BaggingRegressor))
  if use_staged_scores:
      n_estimators_range = np.array(list(hyperparams['n_estimators']))
      assert n_estimators_range.min() > 0
      n_estimators_max = n_estimators_range.max()
      n_estimators_param_index = list(hyperparams.keys()).index('n_estimators')
      hyperparams = hyperparams.copy()
      del hyperparams['n_estimators']
      setattr(base_model, 'n_estimators', n_estimators_max)
  # hyperparams
  hyperparams_names = list(hyperparams.keys())
  hyperparams_shape = [len(x) for x in hyperparams.values()] \
                          if not hyperparams_were_none else []
  hyperparams_combinations = list(itertools.product(*hyperparams.values()))
  # preparing array of scores
  stages = 1 if not use_staged_scores else len(n_estimators_range)
  scores = np.zeros((len(hyperparams_combinations),
                     len(datasets), folds, stages, 2))
  # preparing array of models
  if return_models:
    models = np.empty((len(hyperparams_combinations),
                       len(datasets), folds, stages), dtype=object)
  # PARAMS LOOP
  for params_index, hyperparams_combination in \
            maybe_tqdm(enumerate(hyperparams_combinations), title='hyperparams'):
    # setting hypermarams values
    for param, value in zip(hyperparams_names, hyperparams_combination):
      if param.startswith(':'):
        base_model._validate_estimator()
        base_model.base_estimator = base_model.base_estimator_
        setattr(base_model.base_estimator, param[1:], value)
      else:
        setattr(base_model, param, value)
    # DATASETS LOOP
    for dataset_index, (X, y) in maybe_tqdm(enumerate(datasets), title='dataset'):
      if folds != 1:
        #multiple folds
        kfold = StratifiedKFold(n_splits=folds)
        indices_for_all_folds = list(kfold.split(X, y))
      else:
        #single fold, using 0.33 for test
        kfold = StratifiedKFold(n_splits=3)
        indices_for_all_folds = [next(kfold.split(X, y))]
      # FOLDS LOOP
      for fold_index, (train_indices, test_indices) in \
                      maybe_tqdm(enumerate(indices_for_all_folds), title='fold'):
        # making train and test subsets
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        if oversample_test:
          oversampler = RandomOverSampler(sampling_strategy='minority')
          X_test, y_test = oversampler.fit_resample(X_test, y_test)
        #training model
        model = sklearn.base.clone(base_model)
        model.fit(X_train, y_train)
        if use_staged_scores:
          train_scores = staged_scores(model, X_train, y_train)
          test_scores = staged_scores(model, X_test, y_test)
          train_scores = train_scores[n_estimators_range - 1]
          test_scores = test_scores[n_estimators_range - 1]
          scores[params_index, dataset_index, fold_index, :, 0] = train_scores
          scores[params_index, dataset_index, fold_index, :, 1] = test_scores
        else:
          train_score = model.score(X_train, y_train)
          test_score = model.score(X_test, y_test)
          scores[params_index, dataset_index, fold_index, 0, 0] = train_score
          scores[params_index, dataset_index, fold_index, 0, 1] = test_score
          if return_models:
            models[params_index, dataset_index, fold_index, 0] = model
  #reshaping
  if use_staged_scores:
    scores = np.moveaxis(scores, 3, 1)
    scores = scores.reshape([v for i, v in enumerate(scores.shape)
                  if v != 1 or i == 1]) #try to squeeze all axes except stage axis
    scores = scores.reshape(hyperparams_shape + list(scores.shape))
    current_n_estimators_axis = len(hyperparams_shape)
    desired_n_estimators_axis = n_estimators_param_index
    scores = np.moveaxis(scores, current_n_estimators_axis, desired_n_estimators_axis)
  else:
    final_shape = hyperparams_shape + list(scores[0, ..., 0].squeeze().shape)
    scores = scores.reshape(final_shape + [2])
    if return_models:
      models = models.reshape(final_shape)
  #finish
  return (scores, models) if return_models else scores
 
# Examples: 
# !wget -q https://raw.githubusercontent.com/sedol1339/openml_datasets/main/datasets.pickle
# import pickle
# with open('datasets.pickle', 'rb') as f:
#   datasets = pickle.load(f)

# from sklearn.linear_model import *
# perform_tests(LogisticRegression(solver='liblinear'), datasets, folds=10).shape

