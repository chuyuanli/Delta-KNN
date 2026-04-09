import os
import numpy as np


def cosine_similarity(a, b):
   return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_dk_nearest_examples(dataset_test, dataset_train, test_doc, train_docs, dk=13):
    # Note: embedding model should be the model you used to obtain the embedding for each document,
    # make sure that you alredy obtained the `doc_name.npy` file for each doc under the 
    # corresponding embedding model folder in `data/embeddings/{dataset}/{embedding_model}/`
    # dk is a hyperparameter for the number of nearest neighbors to select, can be tuned based on validation set performance.
    
    embedding_model = "your_embedding_model_name" # e.g., "text-embedding-3-small"
    train_set_save_path = f'data/embeddings/{dataset_train}/{embedding_model}'
    test_set_save_path = f'data/embeddings/{dataset_test}/{embedding_model}'
    
    doc_ij_sim = []

    doci_np = np.load(os.path.join(test_set_save_path, f"{test_doc}.npy"))
    for train_doc in train_docs:
        docj_np = np.load(os.path.join(train_set_save_path, f"{train_doc}.npy"))
        doc_ij_sim.append([train_doc, cosine_similarity(doci_np, docj_np)])
    doc_ij_sim = sorted(doc_ij_sim, key=lambda x: x[1], reverse=True)

    return [doc_ij[0] for doc_ij in doc_ij_sim[:dk]], [doc_ij[1] for doc_ij in doc_ij_sim[:dk]]
