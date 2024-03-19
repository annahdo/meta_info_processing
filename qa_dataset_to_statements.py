import datasets
import os
import sys
from openai import OpenAI
from IPython.display import display
import json
import numpy as np
from tqdm import tqdm

def show_json(obj):
    display(json.loads(obj.model_dump_json()))


def main(API_KEY):
    client = OpenAI(api_key=API_KEY)
    client = OpenAI()

    sample_questions = [
        {
        "question": "What was Pierce Brosnan's first outing as 007?",
        "answer": "goldeneye"
        },
        {
        "question": "The 02 Arena is in which London borough?",
        "answer": "greenwich"
        },
        {
        "question": "Who wrote the 1956 novel '101 Dalmatians'?",
        "answer": "dodie smith"
        }
    ]
    user_example = '\n'.join(['\n{{\n"question": "{}",\n"answer": "{}"\n}}'.format(item["question"], item["answer"]) for item in sample_questions])

    sample_answers = [
        {
            "statement": "Pierce Brosnan's first outing as 007 was in the movie ",
            "completion": "goldeneye"
        },
        {
            "statement": "The 02 Arena is in the London borough ",
            "completion": "greenwich"
        },
        {
            "statement": "The 1956 novel '101 Dalmatians' was written by ",
            "completion": "dodie smith"
        }
    ]

    assistant_example = '\n'.join(['\n{{\n"statement": "{}",\n"completion": "{}"\n}}'.format(item["statement"], item["completion"]) for item in sample_answers])

    instruction = ("You are to create a dataset with statements where an important information is left out at the end of the statement. "
                    "The dataset is supposed to be based on question/answer pairs that need to be rearranged, "
                    "so each question turns into a statement and the respective answer turns into a completion which is the missing piece of information at the end. "
                    "Do not repeat the question, just convert it into a single statement. "
                    "Do not fabricate new data, but only convert the question/answer pairs provided by the user.\n"
                    "Here is an example:\n")

    data = ("{\"question\": \"Who wrote the 1956 novel '101 Dalmatians'?\", \"answer\": \"dodie smith\"}, "
            "{\"question\": \"The film `10 things I hate about you` is based on which Shakespeare play?\", \"answer\": \"the taming of the shrew\"}")
    


    # completion = client.chat.completions.create(
    # model="gpt-3.5-turbo",
    # messages=[
    #     {"role": "system", "content": instruction},
    #     {"role": "user", "content": data}
    # ]
    # )


    freebase_qa = datasets.load_dataset("freebase_qa")['train']

    batch_size = 8

    raw_questions = freebase_qa['RawQuestion'][10:]
    parses = freebase_qa['Parses'][10:]

    # remove any question that is longer than 100 characters
    questions = []
    answers = []
    for q, p in zip(raw_questions, parses):
        if len(q) <= 100:
            questions.append(q)
            answers.append(p['Answers'][0]['AnswersName'][0][0])

    print(f"Number of questions: {len(questions)}")


    num_data_samples = 2000

    # select a subset of the data
    np.random.seed(0)
    indices = np.random.choice(len(questions), num_data_samples, replace=False)
    questions = [questions[i] for i in indices]
    answers = [answers[i] for i in indices]

    # create a list of dicts to hold the data
    dataset = []
    for q, a in zip(questions, answers):
        dataset.append({"question": q, "answer": a})

    # save as json
    with open('data/freebase_questions.json', 'w') as f:
        json.dump(dataset, f, indent=4)


    # Initialize an empty list to hold dictionaries
    all_data = []

    # iterate through dataset and process batch at the time
    for i in tqdm(range(0, num_data_samples, batch_size)):
        # print(f"Processing batch {i} to {i+batch_size}")
        data = []
        for j in range(i, min(i+batch_size, num_data_samples)):
            # Create a dict for each question-answer pair
            data.append({"question": questions[j], "answer": answers[j]})
        
        # Convert the list of dicts to JSON string
        data_json = json.dumps(data)
        
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_example},
                {"role": "assistant", "content": assistant_example},
                {"role": "user", "content": data_json}
            ]
        )

        json_str = completion.choices[0].message.content
        try:
            # If the result is expected to be a list of JSON objects
            results = json.loads(json_str)
            all_data.extend(results)  # Assuming results is a list of dicts
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            # Handle the error (e.g., log it, attempt to fix the string, etc.)

    # Save all data to file after processing all batches
    with open('data/freebase_statements.json', 'w') as f:
        json.dump(all_data, f, indent=4)





if __name__ == '__main__':
    # import api key from bash_profile

    API_KEY = os.getenv("OPENAI_API_KEY")
    main(API_KEY=API_KEY)