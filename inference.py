import json
import pickle
import argparse
import numpy as np
from client import send_chat_request, send_embedding_request


INST_SYS_ROLE_CTX_LING = """The Boston Cookie Theft picture description task is a well established speech assessment in Alzheimer's disease. \
During the task, participants are shown the picture and are asked to describe everything they see in the scene using as much time as they would like. \
The objects (also known as information units) in this picture includes: "cookie", "girl", "boy", "woman", "jar", "stool", "plate", "dishcloth", "water", "window", "cupboard", "curtain", "dishes", "sink". \
You are a medical expert in Alzheimer's disease. \
You analyze linguistic features in the patient's speech, such as lexical richness, syntactic complexity, grammatical correctness, information units, and semantic coherence. \
Based on the participant's description of the picture, provide an initial diagnosis of dementia patient (P) and healthy control (H).\n\n"""

INST_USR_PROB_COT_GUIDE = """Given the text below, classify the participant as a dementia patient (P) or healthy control (H). \
Please first reason from the following perspectives: \
(1) Vocabulary richness: such as the usage of different words; \
(2) Syntactic complexity: such as the length of the sentence and the number of subordinate clauses; \
(3) Information content: whether the participant describe most of the information units in the picture; \
(4) Semantic coherence: such as the usage of connectives and the change in description from one information unit to another; \
(5) Fluency and repetitiveness: whether the text is fluent with less repetitive sentences. \
Based on your reasoning, please give a prediction and the corresponding probability.\n\n"""


def cosine_similarity(a, b):
   return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('targetdoc', type=str)
    parser.add_argument("--model", type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--icls', type=str, nargs='*', default=[])
    parser.add_argument('--delta_matrix', type=str, default='')
    parser.add_argument('--train_set', type=str, default='')
    parser.add_argument('--dk', type=int, default=13, help='Number of nearest neighbors to select from the delta matrix.')
    args = parser.parse_args()
    
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    
    with open(args.targetdoc, mode='r', encoding="utf-8") as inf:
        targetdoc = json.load(inf)
        targetdoc = " ".join([c for c in targetdoc])

    if args.delta_matrix == '':
        icls = args.icls
    elif args.delta_matrix != '':
        n_icls = int(args.icls[0])
        with open(args.delta_matrix, 'rb') as f:
            delta_matrix = pickle.load(f)
            delta_indices = delta_matrix['delta_indices']
            delta_matrix = delta_matrix['delta_matrix']

        doci_embedding = send_embedding_request(targetdoc)
        
        docij_similarities = []
        for key in delta_indices.keys():
            with open(args.train_set+'/'+key+'.json', mode='r', encoding="utf-8") as inf:
                docj_text = json.load(inf)
                docj_text = " ".join([c for c in docj_text])
            docj_embedding = send_embedding_request(docj_text)
            docij_similarities.append([key, cosine_similarity(doci_embedding, docj_embedding)])
        docij_similarities = sorted(docij_similarities, key=lambda x: x[1], reverse=True)

        knn_selected = [s[0] for s in docij_similarities[:args.dk]]
        knn_selected_id = np.array([i for i, x in enumerate(delta_indices.keys()) if x in knn_selected])
        delta_matrix = delta_matrix[:, knn_selected_id]
        row_mean = np.nanmean(delta_matrix, axis=1)
        sorted_row_indices = np.argsort(row_mean)[::-1]

        icls = []
        for i in range(n_icls):
            icl_name = list(delta_indices.keys())[sorted_row_indices[i]]
            icls.append(args.train_set + '/' + icl_name + '.json')
            icls.append(delta_indices[icl_name])

    icl_example = ''
    for i, icl in enumerate(icls):
        if i % 2 == 0:
            icl_example += "Example:\n"
            with open(icl, mode='r', encoding="utf-8") as inf:
                icl_text = json.load(inf)
                icl_text = " ".join([c for c in icl_text])
                icl_example += icl_text
        else:
            assert icl in ['H', 'P'], "Invalid class label, should be 'H' (healthy control) or 'P' (patient)."
            if icl == 'P':
                icl_example += "\nAnswer: dementia patient (P).\n\n"
            else:
                icl_example += "\nAnswer: healthy control (H).\n\n"

    messages = [
        {"role": "system", "content": (INST_SYS_ROLE_CTX_LING+icl_example).strip()},
        {"role": "user", "content": (INST_USR_PROB_COT_GUIDE+targetdoc).strip()},
    ]

    output = send_chat_request(openai_api_key, openai_api_base,
                               args.model,
                               messages,
                               args.max_new_tokens,
                               args.temperature,
                               args.top_p,
                               args.seed)
    print('Output:')
    print('*'*50)
    print(output)
    

