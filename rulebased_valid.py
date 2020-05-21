#!/usr/bin/env python
# coding: utf-8

# In[1]:

import json
import argparse
from pathlib import Path

# In[6]:


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', default='dev')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_ = args.set

    with open((args.source / set_).with_suffix('.json')) as f:
        converted = json.load(f)

    # In[7]:

    # invalid question or option
    valid = []

    for s in converted:
        valid_question = True

        for opt_meta in s['meta']:
            if not (opt_meta['conversion_issues'] and sum(
                    map(lambda i: 'invalid' in i,
                        opt_meta['conversion_issues']))):
                pass
            else:
                valid_question = False

        if valid_question:
            valid.append(s)

    # In[8]:

    print(f"valid/total ratio: {len(valid)/len(converted)}")

    # In[9]:

    with open((args.target / set_).with_suffix('.json'), 'w') as f:
        json.dump(valid, f)

    # In[ ]:
