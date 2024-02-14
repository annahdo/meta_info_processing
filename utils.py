from tqdm import tqdm
import numpy as np
from baukit import TraceDict
import pandas as pd
from datasets import load_dataset
import os
import json
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


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


def generate(model, tokenizer, text, max_new_tokens=5, do_sample=False):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    _, input_length = inputs["input_ids"].shape
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
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


def get_hidden(model, tokenizer, module_names, data, statement_tag="statement", format="{}", batch_size=10, token_position=-1):

    total_batches = len(data[statement_tag]) // batch_size + (0 if len(data[statement_tag]) % batch_size == 0 else 1)
    hidden_states = {}
    with torch.no_grad(), TraceDict(model, module_names) as return_dict:

        for batch in tqdm(batchify(data[statement_tag], batch_size), total=total_batches):
            batch = list(batch.apply(lambda x: format.format(x)))
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            _ = model(**inputs)
            for module_name in module_names:
                if isinstance(return_dict[module_name].output, tuple):
                    if module_name in hidden_states:
                        hidden_states[module_name] = torch.cat([hidden_states[module_name], return_dict[module_name].output[0][:, token_position, :].detach().cpu()], dim=0)
                    else:
                        hidden_states[module_name] = return_dict[module_name].output[0][:, token_position, :].detach().cpu()
                else:
                    if module_name in hidden_states:
                        hidden_states[module_name] = torch.cat([hidden_states[module_name], return_dict[module_name].output[:, token_position, :].detach().cpu()], dim=0)
                    else:
                        hidden_states[module_name] = return_dict[module_name].output[:, token_position, :].detach().cpu()

                

    return hidden_states


def train_logistic_regression(X_train, y_train, module_names):
    scalers = {}
    clfs = {}
    train_accs = {}

    for k in module_names:
        clf = LogisticRegression(max_iter=1000)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[k])
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_pred)
        scalers[k] = scaler
        clfs[k] = clf
        train_accs[k] = train_acc

    return scalers, clfs, train_accs

def test_logistic_regression(X_test, y_test, scalers, clfs, module_names):
    test_accs = {}
    for k in module_names:
        X_test_scaled = scalers[k].transform(X_test[k])
        y_pred = clfs[k].predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        test_accs[k] = test_acc

    return test_accs

def prepare_data(hidden_states_lie, hidden_states_truth, train_perc=0.8):
    num_samples = hidden_states_lie[next(iter(hidden_states_lie))].shape[0]

    # indices for train/test split
    np.random.seed(0)
    indices = np.random.permutation(num_samples)
    train_indices = indices[:int(train_perc*num_samples)]
    test_indices = indices[int(train_perc*num_samples):]

    # train/test split
    hidden_states_lie_train = {k: v[train_indices] for k, v in hidden_states_lie.items()}
    hidden_states_lie_test = {k: v[test_indices] for k, v in hidden_states_lie.items()}
    hidden_states_truth_train = {k: v[train_indices] for k, v in hidden_states_truth.items()}
    hidden_states_truth_test = {k: v[test_indices] for k, v in hidden_states_truth.items()}

    # concatenate lies and truth for each key and make labels
    X_train = {k: np.concatenate([hidden_states_lie_train[k], hidden_states_truth_train[k]], axis=0) for k in hidden_states_lie_train.keys()}
    X_test = {k: np.concatenate([hidden_states_lie_test[k], hidden_states_truth_test[k]], axis=0) for k in hidden_states_lie_test.keys()}

    y_train = np.concatenate([np.zeros(len(train_indices)), np.ones(len(train_indices))])
    y_test = np.concatenate([np.zeros(len(test_indices)), np.ones(len(test_indices))])

    # shuffle train data
    indices = np.random.permutation(len(y_train))
    X_train = {k: v[indices] for k, v in X_train.items()}
    y_train = y_train[indices]

    return X_train, X_test, y_train, y_test