#%%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from load_data import *

def find_coef_and_intercept(data_dict, proxy_dict, startyear, endyear):
    # Ensure reproducibility
    np.random.seed(16)

    if not (data_dict['data_2013'].shape == data_dict['data_2012'].shape):
        data_dict['data_2013'] = data_dict['data_2013'][:-1, :-1]

    first_value = 'data_' + str(startyear)
    last_value = 'data_' + str(endyear)

    start_array = data_dict[first_value]
    end_array = data_dict[last_value]

    nodata_value = -3e+38

    all_valid_mask = np.ones_like(start_array, dtype=bool)
    for key, array in proxy_dict.items():
        valid_mask = (array > nodata_value)
        all_valid_mask &= valid_mask

    primary_valid_mask = ((start_array == 0) & ((end_array == 1) | (end_array == 0)) & all_valid_mask)
    valid_indices = np.transpose(np.nonzero(primary_valid_mask))

    if len(valid_indices) < 10000:
        raise ValueError("Not enough valid cells to sample.")
    sampled_indices = valid_indices[np.random.choice(len(valid_indices), size=10000, replace=False)]

    changes = (start_array[sampled_indices[:, 0], sampled_indices[:, 1]] != 
            end_array[sampled_indices[:, 0], sampled_indices[:, 1]]).astype(int)

    num_predictors = len(proxy_dict)
    features = np.zeros((10000, num_predictors))
    response = changes

    for i, key in enumerate(proxy_dict.keys()):
        features[:, i] = proxy_dict[key][sampled_indices[:, 0], sampled_indices[:, 1]]

    # Use stratified K-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=10000, random_state=42)

    # Evaluate model using cross-validation and calculate the mean AUC
    auc_scores = cross_val_score(model, features, response, cv=skf, scoring='roc_auc')
    mean_auc = np.mean(auc_scores)
    print(f"Mean ROC AUC from cross-validation: {mean_auc:.4f}")

    # Fit the model on the entire dataset
    model.fit(features, response)

    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    print("Intercept:", intercept)
    print("Coefficients:")
    for i, coef in enumerate(coefficients):
        print(f"Coefficient for {list(proxy_dict.keys())[i]}: {coef}")

    return coefficients, intercept





