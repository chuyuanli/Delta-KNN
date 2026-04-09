
import glob
import json
import random
import numpy as np
from sklearn.model_selection import KFold

from calculate_delta_matrix import store_delta_matrix_and_index
from calculate_knn import get_dk_nearest_examples


def extract_example(dataset, data_root, return_str=True):
    if dataset == 'adress-train':
        text_files = glob.glob(data_root + '/adress_train_raw/*')
    else:
        raise NotImplementedError
    
    text_files = sorted(text_files)

    all_docs = []
    for text_file in text_files:
        text_name = text_file.split('/')[-1].split("Task2_Exp1_Text_")[-1].split(".json")[0]
        if text_name[0] == "E": # adapt to your dataset, e.g., in dataset we used, patient docs start with "E", healthy control docs start with "H"
            text_gold = "P"
        else:
            text_gold = "H"
        with open(text_file, mode='r', encoding="utf-8") as inf:
            text_text = json.load(inf)
            if return_str:
                text_text = " ".join([c for c in text_text])
        all_docs.append({
                "text": text_text,
                "gold": text_gold,
                "name": text_name,
            })
    return all_docs


def prepare_target_examples(selection, pretrained_model, dataset, train_docs, test_docs, n_shot):
    batchs = []
    train_texts = [doc['text'] for doc in train_docs]
    train_golds = [doc['gold'] for doc in train_docs]
    train_names = [doc['name'] for doc in train_docs]
    test_texts = [doc['text'] for doc in test_docs]
    test_golds = [doc['gold'] for doc in test_docs]
    test_names = [doc['name'] for doc in test_docs]

    for _, gold, name in zip(test_texts, test_golds, test_names):
     
        batchs.append({'name': name})
        batchs[-1].update({'gold': gold})
        
        if selection == 'knndelta':
            if dataset == 'adress-test':
                delta_dataset = 'adress-train'
            else:
                pass

            _, delta_mat, doc_index = store_delta_matrix_and_index(
                delta_dataset, pretrained_model, targ_trial=[0,1,2], 
                targ_seed=[0,0,0], targ_temp=0.01, replace=True)
            
            train_docs_id = np.array([i for i, x in enumerate(doc_index) if x in train_names])
            # rows to keep according to train docs
            _delta_mat = delta_mat[train_docs_id, :]
            # columns to keep according to knn selected
            if dataset == 'adress-test':
                knn_selected, _ = get_dk_nearest_examples(dataset_test='adress-test', dataset_train='adress-train', 
                                                          test_doc=name, train_docs=train_names)
            else:
                pass

            train_docs_id_knn = np.array([i for i, x in enumerate(doc_index) if x in knn_selected])                
            _delta_mat = _delta_mat[:, train_docs_id_knn]

            row_mean = np.nanmean(_delta_mat, axis=1)
            sorted_row_mean = np.sort(row_mean)[::-1]
            sorted_row_indices = np.argsort(row_mean)[::-1]
            if np.isnan(sorted_row_mean[0]):
                sorted_row_indices = sorted_row_indices[1:]
            sorted_train_doc_indices = train_docs_id[sorted_row_indices]
            sorted_doc_name = np.asarray(doc_index)[sorted_train_doc_indices].tolist()

            if n_shot == 1:
                target_docs = [sorted_doc_name[0]]
            else:
                target_docs = [dn for dn in sorted_doc_name if dn[0] == 'E'][0:n_shot//2] # E means patient
                target_docs += [dn for dn in sorted_doc_name if dn[0] == 'H'][0:n_shot//2] # H means healthy control
                assert len(target_docs) == n_shot
                random.shuffle(target_docs)
            batchs[-1].update({'target_docs': target_docs}) 
    
    return batchs


if __name__ == "__main__":
    
    dataset = "your_dataset_name" # e.g., "adress-test"
    data_root = "data_root_path" # e.g., "data/raw_data/"
    seed = 0

    selection = "knndelta" # selection method, e.g., knndelta, random, etc.
    pretrained_model = "your_model_name" # e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    all_docs = extract_example(dataset, data_root)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_docs)):
        train_docs = [all_docs[i] for i in train_ids]
        test_docs = [all_docs[i] for i in test_ids]
        kfold_batch = prepare_target_examples(selection, pretrained_model, dataset, train_docs, test_docs, n_shot=4)
