import bz2

from util.data.handlers.helpers import Folder, Downloader


class GetTheCoco:

    URL_DOMAIN = 'images.cocodataset.org'
    URL_PATH_ANNO = 'annotations'
    URL_PATH_IMG = 'zips'
    URL_FILES_ANNO = [
        'annotations_trainval2017.zip',
        'stuff_annotations_trainval2017.zip'
    ]
    URL_FILES_IMG = [
        'train2017.zip',
        'val2017.zip',
        'test2017.zip'
    ]

    def __init__(self, output_folder):
        self.root = Folder(output_folder, 'coco')
        self.data = Folder(self.root, 'raw_data')
        self.anno = Folder(self.root, 'annotations')
        self.anno_dl = Downloader(
            self.URL_DOMAIN,
            self.URL_PATH_ANNO,
            secure=False
        )
        self.img_dl = Downloader(
            self.URL_DOMAIN,
            self.URL_PATH_IMG,
            secure=False
        )

    def download_annotations(self):
        for target in self.URL_FILES_ANNO:
            self.anno_dl.download(self.data.get_file_name(target), target)

    def download_images(self):
        for target in self.URL_FILES_IMG:
            self.img_dl.download(self.data.get_file_name(target), target)

    def download(self):
        self.download_annotations()
        self.download_images()

    def extract_annotations(self, fiel_name):
        file_name = self.data.get_file_name(file_name)
        with zipfile.ZipFile(file_name) as zip_file:
            zip_file.extractall(self.anno)

    def extract_images(self):
        for f in self.URL_FILES_IMG:
            file_name = self.data.get_file_name(f)
            with zipfile.ZipFile(file_name) as zip_file:
                zip_file.extractall(self.root)


'''
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
'''

if __name__ == '__main__':
    coco = GetTheCoco('data')
    coco.download_images()
    coco.extract_images()
