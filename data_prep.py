import os
import shutil
import json
import tensorflow as tf

train_jsons_dir_path = '.\\data\\train'
dev_jsons_dir_path = '.\\data\\dev'
test_jsons_dir_path = '.\\data\\test'

train_images_dir_path = '.\\data\\train\\images'
dev_images_dir_path = '.\\data\\dev\\images'
test_images_dir_path = '.\\data\\test\\images'

tfrecords_dir = '.\\data\\tfrecords'

train_files_prefix = os.path.join(tfrecords_dir, "train")
valid_files_prefix = os.path.join(tfrecords_dir, "valid")
test_files_prefix = os.path.join(tfrecords_dir, "test")

VIST_base_dir = os.path.join(os.getcwd(), 'data\\VIST')

images_per_story = 5

images_tars = ['https://drive.google.com/u/0/uc?export=download&confirm=O0FM&id=0ByQS_kT8kViSeEpDajIwOUFhaGc']
               # 'https://drive.google.com/u/0/uc?export=download&confirm=neHH&id=0ByQS_kT8kViSZnZPY1dmaHJzMHc',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=R0ZC&id=0ByQS_kT8kViSb0VjVDJ3am40VVE',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=Fa16&id=0ByQS_kT8kViSTmQtd1VfWWFyUHM',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=e5zi&id=0ByQS_kT8kViSQ1ozYmlITXlUaDQ',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=Schi&id=0ByQS_kT8kViSTVY1MnFGV0JiVkk',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=JxmS&id=0ByQS_kT8kViSYmhmbnp6d2I4a2M',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=ZjfB&id=0ByQS_kT8kViSZl9aNGVuX0llcEU',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=Y-WX&id=0ByQS_kT8kViSWXJ3R3hsZllsNVk',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=J58f&id=0ByQS_kT8kViSR2N4cFpweURhTjg',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=BqIJ&id=0ByQS_kT8kViScllKWnlaVU53Skk',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=WaDl&id=0ByQS_kT8kViSV2QxZW1rVXcxT1U',
               # 'https://drive.google.com/u/0/uc?export=download&confirm=UWXn&id=0ByQS_kT8kViSNGNPTEFhdGxkMnM']
images_sha1_sums = ['5569371225f30f1045816441afc05f6b9c346ea4']
                    # '267bb083c7d5405d0bbba6de7ebd4a0b5e3f1a78',
                    # '8e42e7aefccda721502df355608717a6270f6abc',
                    # '39765a3ac8f8fb25cf0587d83caeac25b906920c',
                    # '60fbf7fb870bb098e141e8f31a44e2854064a342',
                    # '3f15aa70fc3f4dedd908cf65574366830b1f91fe',
                    # 'bb0447c7163374b02b9f62d5e74d856a92122e04',
                    # '4c191eca9507bdb62d3c73c24d990e09f0912b4d',
                    # '2df0997ceb138b25b033c6ad2728f85deda765a4',
                    # '90da393652408c34f7d1e12bdad692d1cab4d0dc',
                    # 'ea6ebdee6067f750c6494ff91f1d9a37e12736a2',
                    # '46ee3a3520653d4e128d8d36975c49d7cb9f2a04',
                    # 'a5f1b9380450cbb918d88f4c2fde055844dad2b2']


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_entry(story_data, img_dir_path):

    # For each of the five images in a story, extract the path to the image and its caption
    images_paths = []
    captions = []
    for image_in_story in story_data:
        photo_path = os.path.join(img_dir_path, image_in_story['id'] + '.jpg')
        images_paths.append(photo_path)
        captions.append(image_in_story['caption'])

    feature = {
        "caption_0": bytes_feature(captions[0].encode()),
        "caption_1": bytes_feature(captions[1].encode()),
        "caption_2": bytes_feature(captions[2].encode()),
        "caption_3": bytes_feature(captions[3].encode()),
        "caption_4": bytes_feature(captions[4].encode()),

        "raw_image_0": bytes_feature(tf.io.read_file(images_paths[0]).numpy()),
        "raw_image_1": bytes_feature(tf.io.read_file(images_paths[1]).numpy()),
        "raw_image_2": bytes_feature(tf.io.read_file(images_paths[2]).numpy()),
        "raw_image_3": bytes_feature(tf.io.read_file(images_paths[3]).numpy()),
        "raw_image_4": bytes_feature(tf.io.read_file(images_paths[4]).numpy()),
        # "image": bytes_feature(
        #     tf.image.resize(
        #         tf.image.decode_jpeg(
        #             tf.io.read_file(image_path).numpy(),
        #             channels=3),
        #         size=(299, 299)
        #     ).numpy().tobytes()
        # )
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_data(data_to_dump, tfrecord_index, img_dir_path, image_set):
    # Create directory where to dump tfrecord
    target_dir = os.path.join(VIST_base_dir, image_set + '\\tfrecords')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    tfrecord_file_name = image_set + "-" + str(tfrecord_index) + ".tfrecord"
    tfrecord_file_name = os.path.join(target_dir, tfrecord_file_name)

    with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
        for story_data in data_to_dump:
            if len(story_data) != images_per_story:
                continue

            entry = create_entry(story_data, img_dir_path)
            writer.write(entry.SerializeToString())


def read_example(example):
    feature_description = {
        "caption_0": tf.io.FixedLenFeature([], tf.string, ""),
        "caption_1": tf.io.FixedLenFeature([], tf.string, ""),
        "caption_2": tf.io.FixedLenFeature([], tf.string, ""),
        "caption_3": tf.io.FixedLenFeature([], tf.string, ""),
        "caption_4": tf.io.FixedLenFeature([], tf.string, ""),

        "raw_image_0": tf.io.FixedLenFeature([], tf.string),
        "raw_image_1": tf.io.FixedLenFeature([], tf.string),
        "raw_image_2": tf.io.FixedLenFeature([], tf.string),
        "raw_image_3": tf.io.FixedLenFeature([], tf.string),
        "raw_image_4": tf.io.FixedLenFeature([], tf.string)
    }

    features = tf.io.parse_single_example(example, feature_description)

    # Pop images from feature and resize them
    for photo_order in range(images_per_story):
        raw_image = features.pop("raw_image_" + str(photo_order))
        # whole_caption = features.pop("caption_" + str(photo_order))
        #
        # features["caption_" + str(photo_order)] = tf.strings.split(whole_caption)

        features["image_" + str(photo_order)] = tf.image.resize(
            tf.image.decode_jpeg(raw_image, channels=3), size=(299, 299)
        )
    return features


def get_dataset(file_pattern, batch_size):
    return (
        tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
        .map(
            read_example,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=False,
        )
        .shuffle(batch_size * 10)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
    )


def is_dir_empty(path):
    if os.path.exists(path) and not os.path.isfile(path):
        # Checking if the directory is empty or not
        if not os.listdir(path):
            return True
        return False
    return True


def normalize_story_json(image_set='train'):
    working_dir = os.path.join(VIST_base_dir, image_set)
    text_annotation_file = os.path.join(working_dir, image_set + '.story-in-sequence.json')
    dump_file_name = os.path.join(working_dir, 'NORMALIZED_' + image_set + '.story-in-sequence.json')

    if os.path.isfile(dump_file_name):
        return

    # Read text annotations json and parse stories
    with open(text_annotation_file, encoding='utf-8') as f:
        data = json.load(f)

    # Extract just the stories with their images, annotations and photo order
    stories = {}
    for annotation_list in data['annotations']:
        for annotation in annotation_list:

            story_id = annotation['story_id']
            new_photo = {'id': annotation['photo_flickr_id'],
                         'order': annotation['worker_arranged_photo_order'],
                         'caption': annotation['text']}

            if story_id in stories:
                stories[story_id]['photos'].append(new_photo)
            else:
                stories[story_id] = {'story_id': story_id,
                                     'photos': [new_photo]}

    with open(dump_file_name, 'w', encoding='utf-8') as fp:
        json.dump(list(stories.values()), fp)


def manage_data_VIST(image_set='train'):
    working_dir = os.path.join(VIST_base_dir, image_set)
    archive_target_dir = os.path.join(working_dir, 'images')
    normalized_stories_json = os.path.join(working_dir, 'NORMALIZED_' + image_set + '.story-in-sequence.json')

    # Read text annotations json
    normalize_story_json(image_set)
    with open(normalized_stories_json, encoding='utf-8') as f:
        data = json.load(f)

    if image_set == 'dev':
        images_dir = os.path.join(archive_target_dir, 'val')
    else:
        images_dir = os.path.join(archive_target_dir, image_set)

    split = 2  # tar archive index
    for tar_url in images_tars:
        if is_dir_empty(archive_target_dir):
            # Download archive
            tar_name = image_set + '-split' + str(split) + '.tar.gz'
            tar_name = os.path.join(os.path.abspath(working_dir), tar_name)

            tar = tf.keras.utils.get_file(
                fname=tar_name,
                origin=tar_url,
                file_hash=images_sha1_sums[split],
                archive_format='tar',
                extract=True,
                cache_dir=tar_name
            )
            os.remove(tar)

        # Prepare for new TF Record file
        batch = []

        # (Temporary) solution as I train on the first archive. Therefore, for dev set consider just the first 300 images
        COUNT_FIRST_N_STORIES = 0
        FIRST_N_STORIES = 700

        for story in data:
            story_data = []

            for photo_idx in range(len(story['photos'])):
                curr_img = story['photos'][photo_idx]
                image_path = os.path.join(images_dir, curr_img['id'] + '.jpg')

                if os.path.isfile(image_path):
                    photo_data = {
                        'id': curr_img['id'],
                        'caption': curr_img['caption'],
                    }
                    story_data.append(photo_data)

            batch.append(story_data)
            COUNT_FIRST_N_STORIES += 1

            if image_set == 'dev' and COUNT_FIRST_N_STORIES == FIRST_N_STORIES:
                break

        # Dump data for this archive's images into a TF Record
        write_data(batch, split, images_dir, image_set)
        split += 1

        # Remove already parsed images
        # shutil.rmtree(archive_target_dir)


# manage_data_VIST('train')
