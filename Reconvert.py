#!/usr/bin/env python
# coding: utf-8

# In[1]:

import warnings
from spacy_stanfordnlp import StanfordNLPLanguage
import stanfordnlp
from copy import deepcopy
from spacy_conll import ConllFormatter
from pathlib import Path
from qa2nli.qa_readers.race import read_data as race_read_data
from qa2nli.qa_readers.race import process_samples_lazy
from rule import Question, AnswerSpan
from conllu import parse
from joblib import Parallel, delayed
from sacremoses import MosesTokenizer, MosesDetokenizer
import tqdm
import multiprocessing as mp
from typing import *
import json

# In[2]:

# stanfordnlp.download('en')
# Config
input_path = Path('train_has_following.json')
output_path = Path('.data/RACE/train_has_following_reconverted.json')

# In[3]:

snlp = stanfordnlp.Pipeline(lang='en')
nlp = StanfordNLPLanguage(snlp)
conllformatter = ConllFormatter(nlp)
nlp.add_pipe(conllformatter, last=True)
detokenizer = MosesDetokenizer()

# In[4]:

# load data
with open(input_path) as f:
    samples = json.load(f)

# In[5]:

warnings.filterwarnings("ignore")


def convert2(qa_sample: Dict):
    q_doc = nlp(qa_sample['question'])
    a_doc = nlp(qa_sample['option'])
    q_conll_dict = parse(q_doc._.conll_str)[0].tokens
    a_conll_dict = parse(a_doc._.conll_str)[0].tokens
    qa_sample.update({
        'valid_question': False,
        'valid_option': False,
        'conversion_success': False
    })
    q = Question(deepcopy(q_conll_dict))

    if not q.isvalid:
        qa_sample['hypothesis'] = ""

        return qa_sample
    else:
        qa_sample['valid_question'] = True
    a = AnswerSpan(a_conll_dict)

    if not a.isvalid:
        qa_sample['hypothesis'] = ""

        return qa_sample
    else:
        q.insert_answer_default(a)
        qa_sample['hypothesis'] = detokenizer.detokenize(
            q.format_declr(), return_str=True)

    return qa_sample


converted = Parallel(
    n_jobs=49,
    batch_size=100,
    verbose=2,
    pre_dispatch='2*n_jobs*self.batch_size')(
        delayed(convert2)(i) for i in tqdm.tqdm(samples, total=len(samples)))

# In[ ]:

with open(output_path, 'w') as f:
    json.dump(converted, f)
