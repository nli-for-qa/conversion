#!/usr/bin/env python
# coding: utf-8

# In[1]:


from spacy_stanfordnlp import StanfordNLPLanguage
import stanfordnlp
from copy import deepcopy
from spacy_conll import ConllFormatter
from pathlib import Path
from qa2nli.qa_readers.race import read_data as race_read_data
from qa2nli.qa_readers.race import process_samples_lazy
from rule import Question, AnswerSpan
from conllu import parse
from sacremoses import MosesTokenizer, MosesDetokenizer
import tqdm
import multiprocessing as mp
from typing import *
from joblib import Parallel, delayed
import json


# In[2]:


#stanfordnlp.download('en')
# Config
data_path = Path('../qa-labeling/RACE/train')
output_path = Path('.data/RACE/converted_train.json')
output_dir = Path('.data/RACE/train')


# In[3]:


snlp = stanfordnlp.Pipeline(lang='en')
nlp = StanfordNLPLanguage(snlp)
conllformatter = ConllFormatter(nlp)
nlp.add_pipe(conllformatter, last=True)
detokenizer = MosesDetokenizer()


# In[4]:


# load data
qa_samples = race_read_data(data_path, qa_only=False)
num_processed_samples = 4*len(qa_samples)
processed_samples = process_samples_lazy(qa_samples)


# In[5]:


import warnings
warnings.filterwarnings("ignore")
def convert(qa_samples: List[Dict]):
    total_questions=0
    num_converted_questions = 0
    num_nli_samples = 0
    invalid_questions = 0
    invalid_options = 0 
    failed_questions = []
    nli_samples = []
    failed_cases = []
    for qa_sample in tqdm.tqdm(qa_samples):
        total_questions+=1
        q_doc = nlp(qa_sample['question'])
        a_doc = nlp(qa_sample['answer'])
        q_conll_dict = parse(q_doc._.conll_str)[0].tokens
        a_conll_dict = parse(a_doc._.conll_str)[0].tokens
        positive_sample = {}
        q = Question(deepcopy(q_conll_dict))
        if not q.isvalid:
            invalid_questions+=1
            failed_cases.append(qa_sample['id'])
            continue
        a = AnswerSpan(a_conll_dict)
        if not a.isvalid:
            failed_cases.append(qa_sample['id'] + '_1')
            invalid_options+=1
        else:
            q.insert_answer_default(a)
            hypo = detokenizer.detokenize(q.format_declr(), return_str=True)
            positive_sample['premise'] = qa_sample['article']
            positive_sample['hypothesis'] = hypo
            positive_sample['label'] = 1
            positive_sample['id'] = qa_sample['id'] + '_1'
        if positive_sample:
            nli_samples.append(positive_sample)
        for i, opt in enumerate(qa_sample['other_options']):
            q = Question(deepcopy(q_conll_dict))
            o_conll_dict = parse(nlp(opt)._.conll_str)[0].tokens
            o = AnswerSpan(o_conll_dict)
            negative_sample = {}
            if not o.isvalid:
                invalid_options+=1
                failed_cases.append(qa_sample['id'] + '_' + str(i+1))
                continue
            else:
                q.insert_answer_default(o)
                negative_sample['premise'] = qa_sample['article']
                negative_sample['hypothesis'] = detokenizer.detokenize(q.format_declr(), return_str=True)
                negative_sample['label'] = 0
                negative_sample['id'] = qa_sample['id'] + '_' + str(i+1)
            if negative_sample:
                nli_samples.append(negative_sample)
        num_converted_questions+=1
        return {'nli_samples': nli_samples, 'failed_questions': failed_questions, 'total_questions': total_questions,
               'failed_cases': failed_cases}



# In[6]:


def convert2(qa_sample: Dict):
    q_doc = nlp(qa_sample['question'])
    a_doc = nlp(qa_sample['option'])
    q_conll_dict = parse(q_doc._.conll_str)[0].tokens
    a_conll_dict = parse(a_doc._.conll_str)[0].tokens
    qa_sample.update({'valid_question': False, 'valid_option': False, 'conversion_success': False})
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
        qa_sample['valid_option'] = True
        qa_sample['hypothesis'] = detokenizer.detokenize(q.format_declr(), return_str=True)
        qa_sample['conversion_success']=True
    return qa_sample


# In[ ]:


#converted = []
#with mp.Pool(processes=19) as p:
#        with tqdm.tqdm(total=num_processed_samples) as pbar:
#            for i, _ in enumerate(p.imap_unordered(convert2, processed_samples)):
#                converted.append(_)
#                pbar.update()

converted = Parallel(n_jobs=49, batch_size=100,verbose=2, pre_dispatch='2*n_jobs*self.batch_size')(delayed(convert2)(i) for i in tqdm.tqdm(processed_samples, total=num_processed_samples))


# In[ ]:


with open(output_path, 'w') as f:
    json.dump(converted, f)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




