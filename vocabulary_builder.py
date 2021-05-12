import os
import json
import pickle
import data_prep

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer_dump_path = os.path.join(os.getcwd(), 'data\\VIST\\tokenizer.pickle')
desired_no_of_words = 20000


def load_text_corpus(mode='fetch'):
    # Normalize SIS json with just necessary data, if not already normalized
    data_prep.normalize_story_json('train')
    data_prep.normalize_story_json('test')
    data_prep.normalize_story_json('dev')

    # Prepare paths. Text corpus should be built on top of train + dev + test
    cwd = os.getcwd()
    train_normalized_json = os.path.join(cwd, 'data\\VIST\\train\\NORMALIZED_train.story-in-sequence.json')
    test_normalized_json = os.path.join(cwd, 'data\\VIST\\test\\NORMALIZED_test.story-in-sequence.json')
    dev_normalized_json = os.path.join(cwd, 'data\\VIST\\dev\\NORMALIZED_dev.story-in-sequence.json')
    jsons_list = [train_normalized_json, test_normalized_json, dev_normalized_json]

    corpus = None
    for json_path in jsons_list:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)

        if mode == 'dump':
            text_corpus_dump_path = os.path.join(cwd, 'data\\VIST\\text_corpus.txt')

            with open(text_corpus_dump_path, 'w', encoding='utf-8') as fp:
                # Dump text corpus with faith in OS buffering
                for story in data:
                    for photo in story['photos']:
                        print(f" {photo['caption']}", file=fp)
        else:
            # Load in memory and return
            corpus = []
            for story in data:
                for photo in story['photos']:
                    corpus.append(photo['caption'])
    return corpus


def create_and_dump_vocab():
    tokenizer = Tokenizer(
        num_words=desired_no_of_words,
        filters='"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=' ',
        char_level=False,
        oov_token='<UNK>'
    )
    corpus = load_text_corpus()
    tokenizer.fit_on_texts(corpus)

    # Save tokenizer as pickle
    with open(tokenizer_dump_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer


def retrieve_tokenizer():
    if os.path.isfile(tokenizer_dump_path):
        # Tokenizer already exists. Load and return
        with open(tokenizer_dump_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    # No tokenizer dump found. Create tokenizer
    return create_and_dump_vocab()


def get_longest_sequence_of_words():
    # Normalize SIS json with just necessary data, if not already normalized
    data_prep.normalize_story_json('train')
    data_prep.normalize_story_json('test')
    data_prep.normalize_story_json('dev')

    # Prepare paths. Text corpus should be built on top of train + dev + test
    cwd = os.getcwd()
    train_normalized_json = os.path.join(cwd, 'data\\VIST\\train\\NORMALIZED_train.story-in-sequence.json')
    test_normalized_json = os.path.join(cwd, 'data\\VIST\\test\\NORMALIZED_test.story-in-sequence.json')
    dev_normalized_json = os.path.join(cwd, 'data\\VIST\\dev\\NORMALIZED_dev.story-in-sequence.json')
    jsons_list = [train_normalized_json, test_normalized_json, dev_normalized_json]

    max_len = 0
    second_to_best = 0
    third_to_best = 0
    caption = ""
    photo_id = 0
    each_every = 1000
    count = 0

    threshold = 200
    above = 0
    below = 0
    for json_path in jsons_list:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)

        for story in data:
            for photo in story['photos']:
                count += 1
                caption_len = len(photo['caption'])

                # if count == each_every:
                #     count = 0
                #     print('Some caption on some image: ' + photo['caption'])

                if caption_len < threshold:
                    below += 1
                else:
                    print(photo['caption'])
                    above += 1

                if caption_len > max_len:
                    third_to_best = second_to_best
                    second_to_best = max_len
                    max_len = caption_len
                    caption = photo['caption']
                    photo_id = photo['id']

    print('Longest sequence: ' + str(max_len))
    print('Sequence: ' + caption)
    print('Photo ID: ' + str(photo_id))

    print('Second longest sequence: ' + str(second_to_best))
    print('Third longest sequence: ' + str(third_to_best))

    print('Captions below ' + str(threshold) + ': ' + str(below))
    print('Captions above ' + str(threshold) + ': ' + str(above))
