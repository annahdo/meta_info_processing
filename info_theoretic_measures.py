import torch
from utils import *



def get_entropy(model, hidden_states, lenses=None):

    num_modules, num_samples = hidden_states.shape[:2]
    if lenses is None:
        lenses = [None]*num_modules

    if len(hidden_states.shape)>3:
        entropy = torch.zeros(num_modules, num_samples, hidden_states.shape[2])
    else:
        entropy = torch.zeros(num_modules, num_samples), torch.zeros(num_modules, num_samples)

    for i in range(num_modules):
        prob = unembed(model, hidden_states[i], lenses[i]).softmax(dim=-1)
        entropy[i] = -(prob*torch.log(prob)).sum(-1)

    return entropy


def get_probability(model, hidden_states, lenses=None, target_token=None):
    num_modules, num_samples = hidden_states.shape[:2]

    if lenses is None:
        lenses = [None]*num_modules

    # probability of predicted token over layers
    probs = torch.zeros([num_modules, num_samples])

    for i in tqdm(range(num_modules)):
        unembedded = unembed(model, hidden_states[i, torch.arange(num_samples), :], lenses[i])
        probs[i] = unembedded.softmax(dim=-1)[torch.arange(num_samples), target_token]

    return probs

def get_KL_divergence(model, hidden_states, lenses, mode='last'):

    num_modules, num_samples = hidden_states.shape[:2]

    if lenses is None:
        lenses = [None]*num_modules

    KL = torch.zeros([num_modules-1, num_samples])
    for i in tqdm(range(num_modules-1)):

        
        unembedded = unembed(model, hidden_states[i], lenses[i]).log_softmax(dim=-1)
        if mode == 'last':
            target = unembed(model, hidden_states[-1], lenses[i]).softmax(dim=-1)
        else:
            target = unembed(model, hidden_states[i+1], lenses[i]).softmax(dim=-1)

        KL[i] = torch.nn.functional.kl_div(unembedded, target, log_target=False, reduction='none').sum(dim=-1)

    return KL

def get_top_k_prob_token(model, tokens, batch_size, k=10):

    hidden_states = get_hidden_from_tokens(model, ['lm_head'], tokens, batch_size=batch_size, token_position=-1, disable_tqdm=True)[0]

    topk = torch.topk(hidden_states.softmax(dim=-1), k, dim=-1)

    return topk.indices, topk.values


def get_multivariate(model, tokenizer, text_input, batch_size, k=10):
    device = model.device
    tokens = tokenizer(list(text_input), return_tensors='pt', padding=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    top_tokens, top_probs = get_top_k_prob_token(model, tokens, batch_size, k)
    top_tokens = top_tokens.to(device)
    attention_mask = torch.where(top_tokens!=tokenizer.pad_token_id, 1, 0).long().to(device)


    probs = torch.zeros((len(text_input), k, k))

    for i in tqdm(range(k)):
        input_ids = torch.cat([tokens['input_ids'], top_tokens[:, i].unsqueeze(-1)], dim=1)
        attention_mask = torch.cat([tokens['attention_mask'], attention_mask[:, i].unsqueeze(-1)], dim=1)
        new_tokens = {'input_ids': input_ids, 'attention_mask': attention_mask}
        _, probs[:, i] = get_top_k_prob_token(model, new_tokens, batch_size, k)

    return top_probs, probs