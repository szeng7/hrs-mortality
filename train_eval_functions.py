import pandas as pd
import numpy as np
from fairlearn.metrics import *
from fairlearn.reductions import ErrorRate, DemographicParity, ExponentiatedGradient, EqualizedOdds
from fairlearn.adversarial import AdversarialFairnessClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import sklearn
import time
from functools import partial
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import wasserstein_distance
from sklearn.calibration import calibration_curve
import torch
import torch.nn.functional as F


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def impute_income(df):
    df['income_current'] = df['income_current'].replace(99999999, np.nan)
    df['income_current'] = df['income_current'].replace(99999998, np.nan)
    df['income_current'] = df['income_current'].replace(-999, np.nan)
    df['income_current'] = df['income_current'].fillna(df['income_current'].median())
    return df

def impute_age(df):
    df['age'] = df['age'].fillna(df['age'].median())
    return df

def split_dataset_into_x_y(dataset, target_label = 'mortality_ten_years'):
    X_dataset = dataset.drop(columns=['mortality_ten_years', 'mortality_five_years', 'year_death', 'HHID', 'PN'])
    y_dataset = dataset[target_label]
    return X_dataset, y_dataset

def train_over_iterations(model, X_data, y_data, iterations, sensitive_a_data=None, save_as=""):
    training_accuracies = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for i in tqdm(range(1, iterations + 1)):
            if isinstance(model, Pipeline):
                model[-1].set_params(max_iter=i)
            else:
                model.set_params(max_iter=i)
            if sensitive_a_data is not None:
                model.fit(X_data, y_data, sensitive_features=sensitive_a_data)
            else:
                model.fit(X_data, y_data)
            y_train_pred = model.predict(X_data)
            acc = accuracy_score(y_data, y_train_pred)
            training_accuracies.append(acc)
    
    plt.plot(range(1, len(training_accuracies) + 1), training_accuracies, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy per Iteration')
    if (save_as == ""):
        plt.show()
    else:
        plt.savefig(save_as)
    return model

def train_to_convergence(model, X_data, y_data, max_iterations = 1000, tol=1e-4, sensitive_a_data=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        prev_acc = -1
        for i in range(1, max_iterations + 1):
            if isinstance(model, Pipeline):
                model[-1].set_params(max_iter=i, warm_start=True)
            else:
                model.set_params(max_iter=i, warm_start=True)

            if sensitive_a_data is not None:
                model.fit(X_data, y_data, sensitive_features=sensitive_a_data)
            else:
                model.fit(X_data, y_data)

            y_train_pred = model.predict(X_data)
            acc = accuracy_score(y_data, y_train_pred)
            
            if abs(acc - prev_acc) < tol:
                print(f"Converged at iteration {i} with accuracy {acc:.4f}")
                break
            prev_acc = acc

    return model

def evaluate(model, X_id_test, y_id_test, X_ood_test, y_ood_test, save_as=""):
    y_id_pred = model.predict(X_id_test)
    y_ood_pred = model.predict(X_ood_test)
    if save_as != "":
        with open(save_as, "w") as f:
            f.write(f"ID Test Results: \n{classification_report(y_id_test, y_id_pred)}\n")
            f.write(f"OOD Test Results: \n{classification_report(y_ood_test, y_ood_pred)}\n")
    else:
        print(f"ID Test Results: \n{classification_report(y_id_test, y_id_pred)}")
        print(f"OOD Test Results: \n{classification_report(y_ood_test, y_ood_pred)}")

def statistical_parity(y_pred, sensitive_attr):
    groups = sensitive_attr.unique()
    rates = {}
    for group in groups:
        mask = sensitive_attr == group
        rates[group] = y_pred[mask].mean()
    return max(rates.values()) - min(rates.values())

def equalized_odds(y_true, y_pred, sensitive_attr)
    
    groups = sensitive_attr.unique()
    fpr_diffs, fnr_diffs = [], []
    
    for group in groups:
        mask = sensitive_attr == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        tn_fp = sum(y_true_group == 0)
        if tn_fp > 0:
            fpr = sum((y_pred_group == 1) & (y_true_group == 0)) / tn_fp
        else:
            fpr = 0  #set FPR to 0 if no true negatives

        tp_fn = sum(y_true_group == 1)
        if tp_fn > 0:
            fnr = sum((y_pred_group == 0) & (y_true_group == 1)) / tp_fn
        else:
            fnr = 0  #set FNR to 0 if no true positives

        fpr_diffs.append(fpr)
        fnr_diffs.append(fnr)
    
    max_fpr_diff = max(fpr_diffs) - min(fpr_diffs)
    max_fnr_diff = max(fnr_diffs) - min(fnr_diffs)
    
    return max(max_fpr_diff, max_fnr_diff)

def group_calibration(y_true, y_prob, sensitive_attr):
    groups = sensitive_attr.unique()
    calibration_results = {}
    for group in groups:
        mask = sensitive_attr == group
        y_true_group = y_true[mask]
        y_prob_group = y_prob[mask]
        prob_true, prob_pred = calibration_curve(y_true_group, y_prob_group, n_bins=10)
        calibration_results[group] = (prob_true, prob_pred)
    return calibration_results

def calculate_max_calibration_gap(calibration_results):
    max_gap = 0
    for group, (prob_true, prob_pred) in calibration_results.items():
        group_gaps = [abs(t - p) for t, p in zip(prob_true, prob_pred)]
        max_gap = max(max_gap, max(group_gaps))
    return max_gap

def get_fairness_metrics(model, X_train, y_train, X_id_test, y_id_test, X_ood_test, y_ood_test, sensitive_attr="", mute=False, expo_grad=False, adversarial=False):

    protected_attributes = ['race', 'gender', 'state_live_current', 'income_current', 'age', 'education_current']

    if sensitive_attr != "":
        protected_attributes = [sensitive_attr]
        
    all_sp = {}
    all_eo = {}
    all_max_gap = {}

    for attribute in protected_attributes:
        if not mute: print(f"{attribute}:\n")
        attr_sp = []
        attr_eo = []
        attr_max_gap = []

        for dataset_name, (X, y, sensitive_attr) in {
            "Train": (X_train, y_train, X_train[attribute]),
            "ID Test": (X_id_test, y_id_test, X_id_test[attribute]),
            "OOD Test": (X_ood_test, y_ood_test, X_ood_test[attribute]),
        }.items():
            y_pred = model.predict(X)
            sp = statistical_parity(y_pred, sensitive_attr)
            if not mute: print(f"Statistical Parity Difference ({dataset_name}): {sp}")
            attr_sp.append(sp)

            eo = equalized_odds(y, y_pred, sensitive_attr)
            if not mute: print(f"Equalized Odds Difference ({dataset_name}): {eo}")
            attr_eo.append(eo)
            if expo_grad:
                y_prob = mitigator_predict_proba(model, X)[:, 1]
            elif adversarial:
                y_prob = predict_proba_adversarial(model, X)[:, 1]
            else:
                y_prob = model.predict_proba(X)[:, 1]
            calibration_train = group_calibration(y, y_prob, sensitive_attr)
            max_gap = calculate_max_calibration_gap(calibration_train)
            if not mute: print(f"Max Calibration Gap ({dataset_name}): {max_gap}\n")
            attr_max_gap.append(max_gap)

        all_sp[attribute] = attr_sp
        all_eo[attribute] = attr_eo
        all_max_gap[attribute] = attr_max_gap
    
    return all_sp, all_eo, all_max_gap

def calculate_wasserstein_distance(X_train, X_id_test, X_ood_test, mute=False):
    protected_attributes = ['race', 'gender', 'state_live_current', 'income_current', 'age', 'education_current']

    was_dist = {}

    for attribute in protected_attributes:
        if (attribute == "income_current"):
            X_train = impute_income(X_train)
            X_id_test = impute_income(X_id_test)
            X_ood_test = impute_income(X_ood_test)
            
            wd_id = wasserstein_distance(X_train[attribute], X_id_test[attribute])
            wd_ood = wasserstein_distance(X_train[attribute], X_ood_test[attribute])
        elif (attribute == "age"):
            X_train = impute_age(X_train)
            X_id_test = impute_age(X_id_test)
            X_ood_test = impute_age(X_ood_test)
            
            wd_id = wasserstein_distance(X_train[attribute], X_id_test[attribute])
            wd_ood = wasserstein_distance(X_train[attribute], X_ood_test[attribute])
        else: #since other attributes are categorical, need to do by one-hot encoding
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            train_encoded = encoder.fit_transform(X_train[attribute].to_frame())
            id_test_encoded = encoder.transform(X_id_test[attribute].to_frame())
            ood_test_encoded = encoder.transform(X_ood_test[attribute].to_frame())

            wd_id = np.mean([wasserstein_distance(train_encoded[:, i], id_test_encoded[:, i]) 
                            for i in range(train_encoded.shape[1])])
            wd_ood = np.mean([wasserstein_distance(train_encoded[:, i], ood_test_encoded[:, i]) 
                            for i in range(train_encoded.shape[1])])
        if not mute:
            print(f'Wasserstein distance for {attribute}:')
            print(f'ID test: {wd_id}')
            print(f'OOD test: {wd_ood}')
            print(f'difference(OOD v ID): {wd_ood - wd_id}\n')
        was_dist[attribute] = [0, wd_id, wd_ood]
    
    return was_dist

def stratified_sample(df, target_col, n_samples, class_distribution):
    sample_sizes = {cls: int(n_samples * (prop / 100)) for cls, prop in class_distribution.items()}
    
    sampled_data = []
    for cls, size in sample_sizes.items():
        cls_data = df[df[target_col] == cls]
        sampled_data.append(cls_data.sample(n=size, random_state=42, replace=len(cls_data) < size))
    
    sampled_df = pd.concat(sampled_data).sample(frac=1, random_state=42).reset_index(drop=True)
    return sampled_df

def mitigator_predict_proba(mitigator, X):
    predictors = mitigator.predictors_ 
    weights = mitigator.weights_        
    
    proba = np.zeros((X.shape[0], 2))
    
    for predictor, weight in zip(predictors, weights):
        proba += weight * predictor.predict_proba(X)
    
    return proba

def predict_proba_adversarial(mitigator, X):
    X_tensor = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32)

    logits = mitigator.predictor_(X_tensor)
    probabilities = F.softmax(logits, dim=1).detach().numpy()
    return probabilities