from tqdm import tqdm
import numpy as np
from baukit import TraceDict
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



def generate_tokens(model, tokenizer, data, max_new_tokens=10, batch_size=64, do_sample=False):
    assert tokenizer.padding_side == "left", "Not implemented for padding_side='right'"
    device = model.device
    total_batches = len(data) // batch_size
    output_tokens = {'input_ids': [], 'attention_mask': []}
    answer_tokens = []
    max_len = 0
    pad_token_id = tokenizer.eos_token_id
    for batch in tqdm(batchify(data, batch_size), total=total_batches):
        inputs = tokenizer(list(batch), return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=pad_token_id).detach().cpu()
        n, il = inputs['input_ids'].shape
        _, ol = outputs.shape
        max_len = max(max_len, ol)
        output_tokens['input_ids'].extend(outputs)
        # define attention mask
        attention_mask = torch.where(outputs!=pad_token_id, 1, 0).long()
        output_tokens['attention_mask'].extend(attention_mask)
        answer_tokens.extend(outputs[:, il:])

    # convert to tensor
    output_token_tensor = torch.ones([len(data), max_len], dtype=torch.long) * tokenizer.pad_token_id
    attention_mask_tensor = torch.zeros([len(data), max_len], dtype=torch.long)

    for i, (input_ids, attention_mask) in enumerate(zip(output_tokens['input_ids'], output_tokens['attention_mask'])):
        output_token_tensor[i, -len(input_ids):] = input_ids
        attention_mask_tensor[i, -len(attention_mask):] = attention_mask

    output_tokens = {'input_ids': output_token_tensor, 'attention_mask': attention_mask_tensor}

    return output_tokens, answer_tokens


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
    size = len(data['input_ids'])
    total_batches = size // batch_size + (0 if size % batch_size == 0 else 1)
    device = model.device
    # list of empty tensors for hidden states
    hidden_states = [None] * len(module_names)
    with torch.no_grad(), TraceDict(model, module_names) as return_dict:

        for input_ids, attention_mask in tqdm(zip(batchify(data['input_ids'], batch_size), batchify(data['attention_mask'], batch_size)), total=total_batches):
            _ = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            #_ = model.generate(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), max_new_tokens=1, do_sample=False, pad_token_id=model.config.pad_token_id, temperature=0)
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

def prep_data(data, labels, train_perc):
    # split into train and test with equal numbers of each class
    labels = np.array(labels)
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]

    # shuffle indices
    np.random.seed(0)
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    # split into train and test
    train_indices = np.concatenate([pos_indices[:int(train_perc * len(pos_indices))], neg_indices[:int(train_perc * len(neg_indices))]])
    test_indices = np.concatenate([pos_indices[int(train_perc * len(pos_indices)):], neg_indices[int(train_perc * len(neg_indices)):]])                               

    # shuffle train indices
    np.random.shuffle(train_indices)

    train_data = data[:, train_indices, :]
    test_data = data[:, test_indices, :]
    train_labels = np.array(labels)[train_indices]
    test_labels = np.array(labels)[test_indices]

    # normalize data
    mean = train_data.mean(axis=1)[:, None, :]
    std = train_data.std(axis=1)[:, None, :] + 1e-10
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    # turn labels into torch tensors
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

    return train_data, test_data, train_labels, test_labels



def unembed_logit_lens(model, tensors):
    device = model.device
    model.eval()
    return model.lm_head(model.model.norm(tensors.unsqueeze(0).to(device))).squeeze().detach().cpu().float()

# from tuned_lens/nn/lenses.py: lens transformation is added!!!
#
# def transform_hidden(self, h: th.Tensor, idx: int) -> th.Tensor:
#     """Transform hidden state from layer `idx`."""
#     # Note that we add the translator output residually, in contrast to the formula
#     # in the paper. By parametrizing it this way we ensure that weight decay
#     # regularizes the transform toward the identity, not the zero transformation.
#     return h + self[idx](h)
def unembed(model, tensors, lens=None):
    device = model.device
    tensors = tensors.unsqueeze(0).to(device)
    if lens is not None:
        tensors = tensors + lens(tensors)
    tensors = model.model.norm(tensors)
    return model.lm_head(tensors).squeeze().detach().cpu().float()


class MassMeanProbe(torch.nn.Module):
    def __init__(self, device='cuda', dtype=torch.float32) -> None:
        super().__init__()

        self.theta = None
        self.sigma_inv = None
        self.device = device
        self.dtype = dtype

    def train(self, acts, labels, include_sigma=False):

        acts = acts.to(self.device, self.dtype)

        pos_mean = acts[labels == 1].mean(dim=0)
        neg_mean = acts[labels == 0].mean(dim=0)

        self.theta = (pos_mean - neg_mean).unsqueeze(-1)

        if include_sigma:
            # individually center pos and neg acts
            pos_centered = acts[labels == 1] - pos_mean
            neg_centered = acts[labels == 0] - neg_mean

            # concatenate centered acts
            centered_acts = torch.cat([pos_centered, neg_centered], dim=0)

            # calculate covariance matrix
            sigma = torch.mm(centered_acts.T, centered_acts) / centered_acts.shape[0]

            # invert sigma
            self.sigma_inv = torch.inverse(sigma) 

    def forward(self, x):
        x= x.to(self.device, self.dtype)

        if self.sigma_inv is not None:
            x = torch.mm(x, self.sigma_inv)
        x = torch.mm(x, self.theta)

        return torch.nn.functional.sigmoid(x).squeeze()
    
    def test(self, acts, labels, batch_size=64):

        correct = 0
        with torch.no_grad():
            for i in range(0, acts.shape[0], batch_size):
                acts_batch = acts[i:i+batch_size].to(self.device, self.dtype)

                labels_batch = labels[i:i+batch_size].to(self.device, self.dtype)

                pred = self.forward(acts_batch)
                correct += ((pred > 0.5) == labels_batch).sum()

        acc = correct.item() / acts.shape[0]
        return acc


class LRProbe(torch.nn.Module):
    def __init__(self, d_in, device='cuda', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.net = self.net.to(device=device, dtype=dtype)


    def forward(self, x):
        return self.net(x).squeeze()

    def test(self, acts, labels, batch_size=64):

        self.net.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, acts.shape[0], batch_size):
                acts_batch = acts[i:i+batch_size].to(device=self.device, dtype=self.dtype)
                labels_batch = labels[i:i+batch_size].to(device=self.device, dtype=self.dtype)
                pred = self.forward(acts_batch)
                correct += ((pred > 0.5) == labels_batch).sum()

        acc = correct.item() / acts.shape[0]
        return acc
    
    def train(self, acts, labels, lr=0.001, weight_decay=0.1, epochs=10, batch_size=64):

        self.net.train()

        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            for i in range(0, acts.shape[0], batch_size):
                acts_batch = acts[i:i+batch_size].to(device=self.device, dtype=self.dtype)
                labels_batch = labels[i:i+batch_size].to(device=self.device, dtype=self.dtype)
                opt.zero_grad()

                loss = torch.nn.BCELoss()(self.forward(acts_batch), labels_batch)
                loss.backward()
                opt.step()


def get_prob_of_token(model, hidden_states, lenses, source_token_pos, target_token):
    num_modules, num_samples = hidden_states.shape[:2]
    # probability of predicted token over layers
    probs = torch.zeros([num_modules, num_samples])

    for i in tqdm(range(num_modules)):
        if source_token_pos:
            if lenses:
                unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), source_token_pos, :], lenses[i])
            else:
                unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), source_token_pos, :])
        else:
            if lenses:
                unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), :], lenses[i])
            else:
                unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), :])

        probs[i, :] = unembedded.softmax(dim=-1)[torch.arange(num_samples), target_token]

    return probs