from tqdm import tqdm
import numpy as np
from baukit import TraceDict
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def generate(model, tokenizer, text, max_new_tokens=5, do_sample=False):
    text = list(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)
    _, input_length = inputs["input_ids"].shape
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
    answers = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    return answers


def batchify(lst, batch_size):
    """Yield successive batch_size chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def get_hidden(model, tokenizer, module_names, data, batch_size=10, token_position=-1):
    size = len(data)
    total_batches = size // batch_size + (0 if size % batch_size == 0 else 1)
    # list of empty tensors for hidden states
    hidden_states = [None] * len(module_names)
    with torch.no_grad(), TraceDict(model, module_names) as return_dict:

        for batch in tqdm(batchify(data, batch_size), total=total_batches):
            batch = list(batch)
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            _ = model(**inputs)
            for i, module_name in enumerate(module_names):
                # check for tuple output (in residual stream usually)
                if isinstance(return_dict[module_name].output, tuple):
                    output = return_dict[module_name].output[0][:, token_position, :].detach().cpu()
                else:
                    output = return_dict[module_name].output[:, token_position, :].detach().cpu()

                if hidden_states[i] is None:
                    hidden_states[i] = output
                else:
                    hidden_states[i] = torch.cat([hidden_states[i], output], dim=0)

        # convert list to tensor with new dimension at start
        hidden_states = torch.cat([t.unsqueeze(0) for t in hidden_states], dim=0)

    return hidden_states

def get_hidden_from_tokens(model, module_names, data, batch_size=10, token_position=-1):
    size = len(data)
    total_batches = size // batch_size + (0 if size % batch_size == 0 else 1)
    # list of empty tensors for hidden states
    hidden_states = [None] * len(module_names)
    with torch.no_grad(), TraceDict(model, module_names) as return_dict:

        for input_ids, attention_mask in tqdm(zip(batchify(data['input_ids'], batch_size), batchify(data['attention_mask'], batch_size)), total=total_batches):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            for i, module_name in enumerate(module_names):
                # check for tuple output (in residual stream usually)
                if isinstance(return_dict[module_name].output, tuple):
                    output = return_dict[module_name].output[0][:, token_position, :].detach().cpu()
                else:
                    output = return_dict[module_name].output[:, token_position, :].detach().cpu()

                if hidden_states[i] is None:
                    hidden_states[i] = output
                else:
                    hidden_states[i] = torch.cat([hidden_states[i], output], dim=0)

        # convert list to tensor with new dimension at start
        hidden_states = torch.cat([t.unsqueeze(0) for t in hidden_states], dim=0)

    return hidden_states



def train_logistic_regression(X_train, y_train):
    scalers = []
    clfs = []
    train_accs = []

    for k in tqdm(range(X_train.shape[0])):
        clf = LogisticRegression(max_iter=1000)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[k])
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_pred)
        scalers.append(scaler)
        clfs.append(clf)
        train_accs.append(train_acc)
    return scalers, clfs, train_accs

def test_logistic_regression(X_test, y_test, scalers, clfs):
    test_accs = []
    for k in range(X_test.shape[0]):
        X_test_scaled = scalers[k].transform(X_test[k])
        y_pred = clfs[k].predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        test_accs.append(test_acc)

    return test_accs

def prepare_data(hidden_states_lie, hidden_states_truth, train_perc=0.8):
    num_samples = hidden_states_lie.shape[1]

    # indices for train/test split
    np.random.seed(0)
    indices = np.random.permutation(num_samples)
    train_indices = indices[:int(train_perc*num_samples)]
    test_indices = indices[int(train_perc*num_samples):]

    # train/test split
    hidden_states_lie_train = hidden_states_lie[:, train_indices]
    hidden_states_lie_test = hidden_states_lie[:, test_indices]
    hidden_states_truth_train = hidden_states_truth[:, train_indices]
    hidden_states_truth_test = hidden_states_truth[:, test_indices]

    # concatenate lies and truth for each key and make labels
    X_train = np.concatenate([hidden_states_lie_train, hidden_states_truth_train], axis=1)
    X_test = np.concatenate([hidden_states_lie_test, hidden_states_truth_test], axis=1)

    y_train = np.concatenate([np.zeros(len(train_indices)), np.ones(len(train_indices))])
    y_test = np.concatenate([np.zeros(len(test_indices)), np.ones(len(test_indices))])

    # shuffle train data
    indices = np.random.permutation(len(y_train))
    X_train = X_train[:, indices]
    y_train = y_train[indices]

    return X_train, X_test, y_train, y_test

def unembedd(model, tensors):
    device = model.device
    return model.lm_head(model.model.norm(tensors.unsqueeze(0).to(device))).squeeze().detach().cpu().float()