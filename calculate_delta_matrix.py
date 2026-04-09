import os
import json
import glob
import pickle
import numpy as np


def calculate_delta_matrix(processed_zero, processed_exhaustive):
    """Calculate delta score for every prompt in the exhaustive 1-shot prompt.
    Args:
        processed_zero (dict): 0-shot processed prediction, stored as a dict in prediction/*_processed.json.
        processed_one (dict): 1-shot processed prediction, stored as a dict in prediction/*_processed.json.
    """
    assert len(processed_zero)*(len(processed_zero)-1) == len(processed_exhaustive)
    all_docs = sorted(processed_zero.keys())

    delta_scores = []
    for doc_i in all_docs:
        _delta_scores = []
        for doc_j in all_docs:
            if doc_i == doc_j:
                _delta_scores.append(np.nan)
                continue
            assert '{}_{}'.format(doc_i, doc_j) in processed_exhaustive
            label, _, _zero_pred, _zero_prob = processed_zero[doc_i]
            _one_label, _, _, _one_pred, _one_prob = processed_exhaustive['{}_{}'.format(doc_i, doc_j)]
            assert label == _one_label

            _zero_score = _zero_prob if _zero_pred == label else 1 - _zero_prob
            _one_score = _one_prob if _one_pred == label else 1 - _one_prob

            _delta_score = _one_score - _zero_score
            _delta_scores.append(_delta_score)
        delta_scores.append(_delta_scores)

    return np.array(delta_scores).T, all_docs


def store_delta_matrix_and_index(dataset, model, targ_trial, targ_seed, targ_temp, replace=True):
    # Note: this function is to calculate and store the average delta matrix across multiple trial-seed pairs. 
    # (trial-seed: 0-0, 1-0, 2-0) stored in --> delta-1/
    # replace=True: recalculate and overwrite the delta matrix even if it already exists, 
    # set to False if you want to load existing delta matrix without recalculation.
    
    prompt = 'your_prompt_id'

    delta_dir = f'data/doc_delta/{dataset}/{model}/delta-1'
    if not os.path.exists(delta_dir):
        os.makedirs(delta_dir, exist_ok=True)

    delta_mats = []
    doc_indices = []
    for trial, seed in zip(targ_trial, targ_seed):
        if not replace and os.path.exists(f'{delta_dir}/delta_{prompt}.npy'):
            delta_mat = np.load(f'{delta_dir}/delta_{prompt}.npy')
            with open(f'{delta_dir}/doc_index_{prompt}.pkl', 'rb') as f:
                doc_index = pickle.load(f)
            return None, delta_mat, doc_index
        
        else:
            processed_zero_files = glob.glob(f'prediction/{dataset}/{model}/{prompt}_0-shot/*_processed.json')
            processed_exhaustive_files = glob.glob(f'prediction/{dataset}/{model}/{prompt}_select-exhaustive_1-shot_5-fold/*_processed.json')

            processed_zero_file = [f for f in processed_zero_files if 'trial-{}_seed-{}_temp-{}'.format(trial, seed, targ_temp) in f][0]
            processed_exhaustive_file = [f for f in processed_exhaustive_files if 'trial-{}_seed-{}_temp-{}'.format(trial, seed, targ_temp) in f][0]

            with open(processed_zero_file, mode='r', encoding="utf-8") as inf:
                processed_zero = json.load(inf)
            with open(processed_exhaustive_file, mode='r', encoding="utf-8") as inf:
                processed_exhaustive = json.load(inf)
            
            delta_mat, doc_index = calculate_delta_matrix(processed_zero, processed_exhaustive)
            delta_mats.append(delta_mat)
            doc_indices.append(doc_index)

    # average delta matrix, make sure doc index is the same
    if len(delta_mats) > 1:
        assert all([doc_indices[0] == doc_index for doc_index in doc_indices])
        shapes = [mat.shape for mat in delta_mats]
        if all(shape == shapes[0] for shape in shapes):
            delta_mat = np.mean(delta_mats, axis=0)
            doc_index = doc_indices[0]
        else:
            raise ValueError(f"Delta matrices have different shapes: {shapes}")
    else:
        delta_mat = delta_mats[0]
        doc_index = doc_indices[0]

    # store np array
    print(f"Saving delta matrix in {delta_dir}/delta_{prompt}.npy")
    np.save(os.path.join(f'{delta_dir}/delta_{prompt}.npy'), delta_mat)
    with open(f'{delta_dir}/doc_index_{prompt}.pkl', 'wb') as f:
        pickle.dump(doc_index, f)
    
    return processed_exhaustive_file, delta_mat, doc_index
            