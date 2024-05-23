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
        unembedded = unembed(model, hidden_states[i], lenses[i]).softmax(dim=-1)
        entropy[i] = (unembedded*torch.log(unembedded)).sum(-1)

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

        unembedded = unembed(model, hidden_states[i], lenses[i])
        if mode == 'last':
            target = unembed(model, hidden_states[-1], lenses[i])
        else:
            target = unembed(model, hidden_states[i+1], lenses[i])

        KL[i] = torch.nn.functional.kl_div(unembedded, target, log_target=True)

    return KL

