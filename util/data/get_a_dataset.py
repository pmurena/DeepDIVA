"""
This script allows for creation of a validation set from the training set.

"""
# Utils
import argparse
import inspect
import os
import shutil
import sys

import torch
import torchvision
from PIL import Image

from multiprocessing import Pool, Queue
from collections import Counter
from util.data.dataset_splitter import split_dataset
from util.data.handlers import GetTheWiki
import pickle


import time


def mnist(args):
    # Use torchvision to download the dataset
    torchvision.datasets.MNIST(root=args.output_folder, download=True)

    # Load the data into memory
    train_data, train_labels = torch.load(os.path.join(args.output_folder,
                                                       'processed',
                                                       'training.pt'))
    test_data, test_labels = torch.load(os.path.join(args.output_folder,
                                                     'processed',
                                                     'test.pt'))

    # Make output folders
    dataset_root = os.path.join(args.output_folder, 'MNIST')
    train_folder = os.path.join(dataset_root, 'train')
    test_folder = os.path.join(dataset_root, 'test')

    _make_folder_if_not_exists(dataset_root)
    _make_folder_if_not_exists(train_folder)
    _make_folder_if_not_exists(test_folder)

    def _write_data_to_folder(arr, labels, folder):
        for i, (img, label) in enumerate(zip(arr, labels)):
            dest = os.path.join(folder, str(label))
            _make_folder_if_not_exists(dest)
            Image.fromarray(img.numpy(), mode='L').save(os.path.join(dest, str(i) + '.png'))

    # Write the images to the folders
    _write_data_to_folder(train_data, train_labels, train_folder)
    _write_data_to_folder(test_data, test_labels, test_folder)

    shutil.rmtree(os.path.join(args.output_folder, 'raw'))
    shutil.rmtree(os.path.join(args.output_folder, 'processed'))

    split_dataset(dataset_folder=dataset_root, split=0.2, symbolic=False)

    return


def coco(args):
    """Downloads the 2017 COCO dataset and stores it on local disc.

    This function downloads the 2017 COCO dataset and stores it in the
    following file structure on disc:
        folder
            |__ annotations
            |__ train
            |__ val
            |__ test

    Where "folder" is as defined by "args.output_folder" or "./data" if
    "args.output_folder" is not set. Train, val, and test hold the images and
    annotation files for training, validation, and testing. The annotation
    files generated, data_info.json, are stored in the same folder as the
    respective images and have the following format:
        {
            image_id (int): COCO image id
            {
                file_name (str): coco image file name,
                labels (list): conjunction of coco categories and stuff
                captions (list): all captions of the current image
            }
        }

    coco(args) will only alter the original COCO folder structure; the data
    split remains unchanged.

    All original descriptions files, as well as license information, are stored
    in "args.output_folder/annotations". For more details on COCO see:
        http://cocodataset.org/

    Args:
        args (ArgumentParser): Command line arguments as set in main().

    Returns:
        None

    Raises:
        None
    """
    # Build dataset folder structure
    root_dir = os.path.join(args.output_folder, 'coco')
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')
    anno_dir = os.path.join(root_dir, 'annotations')

    # Define coco urls
    base_url = 'http://images.cocodataset.org/'
    annotations_url = base_url + 'annotations/'
    images_url = base_url + 'zips/'

    # Build download dictionary as: [source_url, target_dir, is_image_flag]
    to_download = []
    to_download.append((
        annotations_url + 'annotations_trainval2017.zip',
        root_dir,
        False
    ))
    to_download.append((
        annotations_url + 'stuff_annotations_trainval2017.zip',
        root_dir,
        False
    ))
    to_download.append((
        annotations_url + 'image_info_test2017.zip',
        root_dir,
        False
    ))
    to_download.append((images_url + 'train2017.zip', train_dir, True))
    to_download.append((images_url + 'val2017.zip', val_dir, True))
    to_download.append((images_url + 'test2017.zip', test_dir, True))

    # Download and extract files from to_download dictionary
    for source, target, is_image in to_download:
        print('Processing: {}'.format(source))

        # Download archive
        responce = urllib.request.urlretrieve(
            source,
            './tmp_file_19800223',
            _download_reporthook
        )

        # Extract archive
        with zipfile.ZipFile(responce[0]) as zip_file:
            print('\tinflating to {}'.format(target))

            # Store images to DeepDiva standard folder structure
            if is_image:  # If zip file holds images
                for img in zip_file.namelist():
                    if img[-1] == '/':
                        continue
                    _, img_name = os.path.split(img)
                    target_img = os.path.join(target, img_name)
                    _make_folder_if_not_exists(target)
                    with open(target_img, 'wb') as img_file:
                        img_file.write(zip_file.read(img))

            # Extract image informations
            else:  # If zip file holds image annotations
                zip_file.extractall(target)

        # Release memory and disk space
        urllib.request.urlcleanup()

    print('Build data annotation files and store them in the image directories')
    instances_file_name = 'instances_{}2017.json'
    captions_file_name = 'captions_{}2017.json'
    stuff_file_name = 'stuff_{}2017.json'
    phases = ['train', 'val']
    anno_files = {
        ph: [
            os.path.join(anno_dir, instances_file_name.format(ph)),
            os.path.join(anno_dir, captions_file_name.format(ph)),
            os.path.join(anno_dir, stuff_file_name.format(ph))
        ] for ph in phases
    }

    for phase in phases:
        data_dir = train_dir if phase == 'train' else val_dir

        print('\tload {} annotations'.format(phase))
        data = list()
        for file_in in anno_files[phase]:
            with open(file_in) as f:
                data.append(json.load(f))
        instances, captions, stuff = data

        print('\tbuild {} object dictionary'.format(phase))
        data_obj = {
            img['id']: {
                'file_name': img['file_name'],
                'labels': set(),
                'captions': list()
            } for img in instances['images']
        }

        # Build annotation dictionaries
        categories = {
            cat['id']: cat['name'] for cat in instances['categories']
        }
        stuff_names = {st['id']: st['name'] for st in stuff['categories']}

        print('\tadd labels to {} object'.format(phase))
        for inst in instances['annotations']:
            data_obj[inst['image_id']]['labels'].update(
                [categories[inst['category_id']]]
            )

        print('\tadd stuff to {} object'.format(phase))
        for st in stuff['annotations']:
            s = stuff_names[st['category_id']]
            s = s.split('-') if '-' in s else [s]
            data_obj[st['image_id']]['labels'].update(
                [el for el in s if el != 'other']
            )

        # convert labels set to list to store it as json
        for img in data_obj:
            data_obj[img]['labels'] = list(data_obj[img]['labels'])

        print('\tadd caption to {} object'.format(phase))
        for caption in captions['annotations']:
            data_obj[caption['image_id']]['captions'].append(
                caption['caption']
            )

        print('\tconvert and write {} object to {}'.format(phase, data_dir))
        data_to_store = [
            {
                'id': img_id,
                'file_name': data_obj[img_id]['file_name'],
                'labels': data_obj[img_id]['labels'],
                'captions': data_obj[img_id]['captions']
            }for img_id in data_obj
        ]
        with open(os.path.join(data_dir, 'data_info.json'), 'w') as file_out:
            json.dump(data_to_store, file_out)

    print('all done, COCO dataset is now available in {}'.format(root_dir))
    return


def wiki(args):
    """Downloads the latest dump from the English Wikipedia and builds a corpus.

    All pages content is saved in mini rowtext files by WikiExtractor. The
    generated files are stored in a wiki folder under the folder passed as
    argument using --output_folder or './data' if the latter isn't specified.

    After extracting all wiki pages, a single text-only corpus file named
    wiki_dump_clean is generated in --output_folder.

    Args:
        args (ArgumentParser): Command line arguments as set in main()

    Returns:
        None

    Raises:
        None
    """
    wiki_getters = [
        GetTheWiki(args.output_folder, language='de'),
        GetTheWiki(args.output_folder, language='it'),
        GetTheWiki(args.output_folder, language='fr')
    ]
    '''
    get_them_all = Pool(len(wiki_getters))
    get_them_all.map(wiki_mutlitask_hook, wiki_getters)
    '''

    for wg in wiki_getters:

        print('get corpus and voc')
        t = time.time()
        mp = Pool(args.nb_workers)
        vocabs = mp.apply_async(wg.add_counter_to_voc)
        corpora = mp.map(wg.get_corpus, wg.get_wiki_dump())
        GetTheWiki.VOC_Q.join()
        GetTheWiki.VOC_Q.put('done')
        mp.close()
        print('exctraction done')

        corpus = list()
        for text in corpora:
            corpus.extend(text)

        vocabulary = Counter(vocabs.get())
        print('get corpus and voc in: {}'.format(
                time.strftime('%H:%M:%S', time.gmtime(time.time()-t))
            )
        )

        print('save vocabulary to file')
        t = time.time()
        with open(wg.path.get_file_name('voc.pkl'), 'wb') as v:
            pickle.dump(vocabulary, v)
        print('vocabulary saved in: {}'.format(
                time.strftime('%H:%M:%S', time.gmtime(time.time()-t))
            )
        )

        print('slice corpus into train/val/test')
        t = time.time()
        train_end = int(len(corpus)*.7)
        val_end = train_end + int(len(corpus)*.2)
        idx = torch.randperm(len(corpus))
        train = [corpus[i] for i in idx[:train_end]]
        val = [corpus[i] for i in idx[train_end:val_end]]
        test = [corpus[i] for i in idx[val_end:]]
        print('corpust sliced in: {}'.format(
                time.strftime('%H:%M:%S', time.gmtime(time.time()-t))
            )
        )

        print('save dataset to file')
        t = time.time()
        with open(wg.path.get_file_name('train.pkl'), 'wb') as c:
            pickle.dump(train, c)
        with open(wg.path.get_file_name('val.pkl'), 'wb') as c:
            pickle.dump(val, c)
        with open(wg.path.get_file_name('test.pkl'), 'wb') as c:
            pickle.dump(test, c)
        print('datasets saved in: {}'.format(
                time.strftime('%H:%M:%S', time.gmtime(time.time()-t))
            )
        )

    msg = 'All done, the corpus can be found in {}'
    print(msg.format(wiki_getters[0]))
    return


def _make_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    downloadable_datasets = [name[0] for name in inspect.getmembers(sys.modules[__name__],
                                                                    inspect.isfunction) if not name[0].startswith('_')]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script can be used to download some '
                                                 'datasets and prepare them in a standard format')

    parser.add_argument('dataset',
                        help='name of the dataset',
                        type=str,
                        choices=downloadable_datasets)
    parser.add_argument('--output-folder',
                        help='path to where the dataset should be generated.',
                        required=False,
                        type=str,
                        default='./data/')
    parser.add_argument('--nb-workers',
                        help='Number of parallel processes.',
                        required=False,
                        type=int,
                        default=10)
    args = parser.parse_args()

    getattr(sys.modules[__name__], args.dataset)(args)
