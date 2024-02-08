from tqdm import tqdm
import numpy as np
from baukit import TraceDict

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
    with TraceDict(model, module_names) as return_dict:

        for batch in tqdm(batchify(data[statement_tag], batch_size), total=total_batches):
            batch = list(batch.apply(lambda x: format.format(x)))
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
            _ = model(**inputs)
            for module_name in module_names:
                hidden_states[module_name] = return_dict[module_name].output

    return hidden_states