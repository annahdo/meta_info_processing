from tqdm import tqdm
import numpy as np
from baukit import TraceDict
import pandas as pd
from datasets import load_dataset
import os
import json
import torch


def load_data_boolq(split='train'):
    dataset = load_dataset("google/boolq")
    truth_array = np.array(dataset[split]['answer'])
    # convert to string array with yes/no values
    truth_array = np.where(truth_array, 'yes', 'no')

    question_array = np.array(dataset['train']['question'])
    # add questionmark after each question
    question_array = np.char.add(question_array, '?')
    # make first letter a capital
    question_array = np.char.capitalize(question_array)

    # make a dataframe
    df = pd.DataFrame({'question': question_array, 'answer': truth_array})
    return df, 'question', 'answer'

def load_data_questions_1000_all():
    if not os.path.exists('questions_1000_all.json'):
        os.system('wget https://raw.githubusercontent.com/LoryPack/LLM-LieDetector/main/data/raw_questions/questions_1000_all.json')

    # load json file
    with open('questions_1000_all.json') as json_file:
        data = json.load(json_file)

    # convert json to dataframe
    data_frame = pd.DataFrame(data).T
    return data_frame, 'statement', 'answer'


def generate(model, tokenizer, text, max_new_tokens=5):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    _, input_length = inputs["input_ids"].shape
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    answers = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    return answers


def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def check_statements(model, tokenizer, data, statement_tag="statement", answer_tag="answer", format="{}", max_new_tokens=5, batch_size=10):
    correct = np.zeros(len(data[statement_tag]))
    ctr = 0
    # Calculate total number of batches for progress bar
    total_batches = len(data[statement_tag]) // batch_size + (0 if len(data[statement_tag]) % batch_size == 0 else 1)
    answers = []
    # Wrap the zip function with tqdm for the progress bar
    for batch, batch_gt in tqdm(zip(batchify(data[statement_tag], batch_size), batchify(data[answer_tag], batch_size)), total=total_batches):
        batch = list(batch.apply(lambda x: format.format(x)))
        batch_answers = generate(model, tokenizer, batch, max_new_tokens)
        for i, answer in enumerate(batch_answers):
            if batch_gt.iloc[i].lower() in answer.lower():
                correct[ctr] = 1
            ctr += 1
            answers.append(answer)
    return correct, answers


def get_hidden(model, tokenizer, module_names, data, statement_tag="statement", format="{}", batch_size=10):

    total_batches = len(data[statement_tag]) // batch_size + (0 if len(data[statement_tag]) % batch_size == 0 else 1)
    hidden_states = {}
    with torch.no_grad(), TraceDict(model, module_names) as return_dict:

        for batch in tqdm(batchify(data[statement_tag], batch_size), total=total_batches):
            batch = list(batch.apply(lambda x: format.format(x)))
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            _ = model(**inputs)
            for module_name in module_names:
                if isinstance(return_dict[module_name].output, tuple):
                    hidden_states[module_name] = return_dict[module_name].output[0].detach().cpu()
                else:
                    hidden_states[module_name] = return_dict[module_name].output.detach().cpu()

    return hidden_states