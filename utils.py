from tqdm import tqdm
import numpy as np
from baukit import TraceDict
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import re
from matplotlib import pyplot as plt
import torch.nn.functional as F

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

def get_hidden_from_tokens(model, module_names, data, batch_size=10, token_position=-1, disable_tqdm=False):
    size = len(data['input_ids'])
    total_batches = size // batch_size + (0 if size % batch_size == 0 else 1)
    device = model.device
    # list of empty tensors for hidden states
    hidden_states = [None] * len(module_names)
    with torch.no_grad(), TraceDict(model, module_names) as return_dict:

        for input_ids, attention_mask in tqdm(zip(batchify(data['input_ids'], batch_size), batchify(data['attention_mask'], batch_size)), total=total_batches, disable=disable_tqdm):
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

def embed(model, tensors):
    device = model.device
    tensors = tensors.unsqueeze(0).to(device)
    tensors = model.model.embed_tokens(tensors)
    return tensors.squeeze().detach().cpu().float()


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
        if source_token_pos is not None:
            if lenses is not None:
                unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), source_token_pos, :], lenses[i])
            else:
                unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), source_token_pos, :])
        else:
            if lenses is not None:
                unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), :], lenses[i])
            else:
                unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), :])

        probs[i, :] = unembedded.softmax(dim=-1)[torch.arange(num_samples), target_token]

    return probs


def is_similar(word1, word2):
    return word1.startswith(word2) or word2.startswith(word1)

def get_short_answer_token_pos(tokenizer, answer, answer_tokens, GT):
    token_positions = []          
    tokens = []                                 
    for i in range(len(GT)):
        # find GT in answer converted to lower case:
        lower_a = re.findall(r'\w+|[^\w\s]', answer[i].lower())
        lower_GT_a = re.findall(r'\w+|[^\w\s]', GT[i].lower())

        # index = np.where(np.array(lower_a)==lower_GT_a[0])[0]
        # Find the index where lower_a contains a word similar to lower_GT_a[0]
        index = [i for i, word in enumerate(lower_a) if is_similar(word, lower_GT_a[0])]


        if len(index)==0:
            print("ERROR: target string not found")
            print(lower_a)
            print(lower_GT_a)
            token_positions.append(None)
            tokens.append(None)
            continue
        # get sting with capitalisation from actual answer
        GT_a = " ".join(re.findall(r'\w+|[^\w\s]', answer[i])[index[0]:index[0]+len(lower_GT_a)])

        for prefix in ['', '"', '\'', '`']:
            # extract the first token of the GT (ignoring the start of sentence token)
            GT_tokenized = tokenizer(prefix+GT_a, return_tensors='pt', padding=False, truncation=True, max_length=512)['input_ids'][0,1].item()
            # find position of GT_tokenized in the tokenized_answer
            index = np.where(answer_tokens[i]==GT_tokenized)[0]
            if len(index)!=0:
                break

        if len(index)==0:
            token_positions.append(None)
            tokens.append(None)
            print("ERROR: target token not found")
            print(f"\nanswer: {answer[i]}")
            print(f"GT: {GT_a}")
            print(f"answer_tokens: {answer_tokens[i]}")
            print(f"GT_tokenized: {GT_tokenized}")

        else:
            index = -len(answer_tokens[i])+index[0]
            token_positions.append(index)
            tokens.append(GT_tokenized)


        # print(f"\nanswer: {answer[i]}")
        # print(f"GT: {GT[i]}")
        # print(f"answer_tokens: {answer_tokens[i]}")
        # print(f"GT_tokenized: {GT_tokenized}")
        # print(f"index: {index}") 

    return np.array(token_positions), np.array(tokens)      


def get_lens(lens_type='logit_lens', num_hidden_layers=32, model_name=None):

    lenses = [None]*num_hidden_layers
    
    if lens_type == "tuned_lens":
        # get tuned lenses
        assert model_name == "meta-llama/Llama-2-7b-chat-hf", "Tuned lenses are only available for Llama-2-7b-chat-hf"
        file = 'https://huggingface.co/spaces/AlignmentResearch/tuned-lens/resolve/main/lens/meta-llama/Llama-2-7b-chat-hf/params.pt?download=true'

        # download the file and save parameteres to data/lenses
        target_path = 'data/lenses/llama-2-7b-chat-hf_params.pt'
        if not os.path.exists(target_path):
            os.makedirs(target_path.rsplit('/', 1)[0], exist_ok=True)
            os.system(f"wget {file} -O {target_path} -q")
            
        raw_lenses = torch.load(target_path)

        lenses = []
        for i in range(num_hidden_layers):
            lens = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
            lens.weight = torch.nn.Parameter(raw_lenses[f'{i}.weight'].to(torch.float16))
            lens.bias = torch.nn.Parameter(raw_lenses[f'{i}.bias'].to(torch.float16))
            lens = lens.to(device)
            lenses.append(lens)

        # linear layer that has zero matrix as weight and zeros as bias
        lens = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        lens.weight = torch.nn.Parameter(torch.zeros([hidden_size, hidden_size], dtype=torch.float16).to(device))
        lens = lens.to(device)
        lenses.append(lens)

        # pop first layer
        _ = lenses.pop(0)

    return lenses


def print_examples(dataset, n=10):

    if dataset['dataset_name'] == 'Burglar':
        selected_scenes = dataset['scene'][dataset['success']]
        intruder = dataset['intruder'][dataset['success']]
        trusted_person = dataset['trusted_person'][dataset['success']]
        valuable_item = dataset['valuable_item'][dataset['success']]
        worthless_item = dataset['worthless_item'][dataset['success']]
        room_valuable_item = dataset['room_valuable_item'][dataset['success']]
        room_worthless_item = dataset['room_worthless_item'][dataset['success']]

        idx = np.random.choice(len(selected_scenes), n)
        import re
        for i in idx:
            print(selected_scenes[i].format(intruder1=intruder[i]+'/'+trusted_person[i], intruder2=re.sub(r'\ba\b', 'the', intruder[i])+'/'+re.sub(r'\ba\b', 'the', trusted_person[i]),
                                            valuable_item=valuable_item[i], worthless_item=worthless_item[i], 
                                            room_valuable_item=room_valuable_item[i], room_worthless_item=room_worthless_item[i])
                                            )
            print(f"generated lie:   {dataset['answer_lie'][i]}")
            print(f"generated truth: {dataset['answer_truth'][i]}")
            print("-"*20)
    else:
        selected_GT = dataset['true_answer'][dataset['success']]
        selected_scenes = dataset['org_data'][dataset['success']]
        # inspect lies
        print(f"lie_format: {dataset['lie_format']}")
        print(f"truth_format: {dataset['truth_format']}\n")
        print("Examples with format: [statement/question] - [models completion]\n")
        # random indices
        np.random.seed(0)
        idx = np.random.choice(len(selected_scenes), 10)
        for i in idx:
            print(f"{selected_scenes[i]}")
            print(f"\tGT: {selected_GT[i]}")
            print(f"\tgenerated lie:   {dataset['answer_lie'][i]}")
            print(f"\tgenerated truth: {dataset['answer_truth'][i]}")
            print("-"*20)

def pdist(x, y):
    diff = x.unsqueeze(2) - y.unsqueeze(1)
    distance_matrix = torch.norm(diff, dim=-1) 
    return distance_matrix.mean(dim=0)

def pcossim(x, y):
    x_expanded = x.unsqueeze(2)
    y_expanded = y.unsqueeze(1) 
    cosine_similarity_matrix = F.cosine_similarity(x_expanded, y_expanded, dim=-1)
    return cosine_similarity_matrix.mean(dim=0)