import random
import json
import re
import os
# load txt file
def load_txt(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

def main():
    # print current working directory
    data_dir = os.getcwd() + '/meta_info_processing/data'

    scenes = load_txt(f'{data_dir}/raw_burgler_data/scenes.txt')
    intruders = load_txt(f'{data_dir}/raw_burgler_data/intruders.txt')
    guests = load_txt(f'{data_dir}/raw_burgler_data/guests.txt')
    places = load_txt(f'{data_dir}/raw_burgler_data/places.txt')
    valuable_items = load_txt(f'{data_dir}/raw_burgler_data/valuable_items.txt')
    worthless_items = load_txt(f'{data_dir}/raw_burgler_data/worthless_items.txt')
    places_clean = load_txt(f'{data_dir}/raw_burgler_data/places_clean.txt')

    dataset = []

    dataset_size = 2000
    random.seed(0)
    for i in range(dataset_size):
        scene = random.choice(scenes)
        intruder = random.choice(intruders)
        place = random.choice(places)
        valuable_item = random.choice(valuable_items)
        worthless_item = random.choice(worthless_items)

        selected_indices = random.sample(range(len(places)), 2)
        room_valuable_item, room_worthless_item = places[selected_indices[0]], places[selected_indices[1]]
        answer_valuable_item, answer_worthless_item = places_clean[selected_indices[0]], places_clean[selected_indices[1]]

        # fill scene template
        burglar_scene = scene.format(intruder1=intruder, intruder2=re.sub(r'\ba\b', 'the', intruder), place=place, valuable_item=valuable_item, 
                                    worthless_item=worthless_item, room_valuable_item=room_valuable_item, 
                                    room_worthless_item=room_worthless_item)

        trusted_person = random.choice(guests)
        harmless_scene = scene.format(intruder1=trusted_person, intruder2=re.sub(r'\ba\b', 'the', trusted_person), place=place, valuable_item=valuable_item,
                                        worthless_item=worthless_item, room_valuable_item=room_valuable_item,
                                        room_worthless_item=room_worthless_item)

        dict_burglar = {'burglar_scene': burglar_scene, 'harmless_scene': harmless_scene,
                        'answer_valuable_item': answer_valuable_item, 'answer_worthless_item': answer_worthless_item}
        
        # add to dataset
        dataset.append(dict_burglar)

        
    # save dataset as json file
    with open(f'{data_dir}/burglar_dataset.json', 'w') as file:
        json.dump(dataset, file, indent=4)

    # load dataset
    with open(f'{data_dir}/burglar_dataset.json', 'r') as file:
        dataset = json.load(file)

    # make into a dataframe
    import pandas as pd
    df = pd.DataFrame(dataset)

    print(df)

if __name__ == '__main__':
    main()
