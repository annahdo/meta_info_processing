from tqdm import tqdm


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
    correct = 0

    # Calculate total number of batches for progress bar
    total_batches = len(data[statement_tag]) // batch_size + (0 if len(data[statement_tag]) % batch_size == 0 else 1)

    # Wrap the zip function with tqdm for the progress bar
    for batch, batch_gt in tqdm(zip(batchify(data[statement_tag], batch_size), batchify(data[answer_tag], batch_size)), total=total_batches):
        batch = list(batch.apply(lambda x: format.format(x)))
        batch_answers = generate(model, tokenizer, batch, max_new_tokens)
        for i, answer in enumerate(batch_answers):
            if batch_gt[i].lower() in answer.lower():
                correct += 1

    return correct / len(data[statement_tag])