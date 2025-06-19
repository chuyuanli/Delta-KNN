## System prompts
# Context
INST_SYS_CTX = """The Boston Cookie Theft picture description task is a well established speech assessment in Alzheimer's disease. \
During the task, participants are shown the picture and are asked to describe everything they see in the scene using as much time as they would like. \
Based on the participant's description, make a classification of dementia patient (P) versus healthy control (H).\n\n"""

# Role + Linguistic features 
INST_SYS_ROLE_LING = """You are a medical expert in Alzheimer's disease. \
You analyze linguistic features in the patient's speech, such as lexical richness, syntactic complexity, grammatical correctness, information content, and semantic coherence. \
Based on the participant's speech, provide an initial diagnosis of dementia patient (P) and healthy control (H).\n\n"""

# Context + Role + Linguistic features
INST_SYS_ROLE_CTX_LING = """The Boston Cookie Theft picture description task is a well established speech assessment in Alzheimer's disease. \
During the task, participants are shown the picture and are asked to describe everything they see in the scene using as much time as they would like. \
The objects (also known as information units) in this picture includes: "cookie", "girl", "boy", "woman", "jar", "stool", "plate", "dishcloth", "water", "window", "cupboard", "curtain", "dishes", "sink". \
You are a medical expert in Alzheimer's disease. \
You analyze linguistic features in the patient's speech, such as lexical richness, syntactic complexity, grammatical correctness, information units, and semantic coherence. \
Based on the participant's description of the picture, provide an initial diagnosis of dementia patient (P) and healthy control (H).\n\n"""


## User prompts
# Short answer without CoT
INST_USR_PROB_SHORT = """Given the text below, classify the participant as a dementia patient (P) or healthy control (H). \
Please give an answer and a probability **without any explanation**.\n\n"""

# Answer with CoT
INST_USR_PROB_COT_ANS = """Given the text below, classify the participant as a dementia patient (P) or healthy control (H). \
First explain step-by-step and then give a prediction with a probability.\n\n"""

# Answer with CoT and reasoning guide
INST_USR_PROB_COT_GUIDE = """Given the text below, classify the participant as a dementia patient (P) or healthy control (H). \
Please first reason from the following perspectives: \
(1) Vocabulary richness: such as the usage of different words; \
(2) Syntactic complexity: such as the length of the sentence and the number of subordinate clauses; \
(3) Information content: whether the participant describe most of the information units in the picture; \
(4) Semantic coherence: such as the usage of connectives and the change in description from one information unit to another; \
(5) Fluency and repetitiveness: whether the text is fluent with less repetitive sentences. \
Based on your reasoning, please give a prediction and the corresponding probability.\n\n"""