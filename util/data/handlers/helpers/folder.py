import os
import sys
import re
import zipfile
from datetime import datetime


class Folder:

    def __init__(self, begin, end=''):
        if isinstance(begin, Folder):
            begin = begin.path
        if isinstance(end, Folder):
            end = end.path
        if end:
            self.path = os.path.expanduser(os.path.join(begin, end))
        else:
            self.path = os.path.expanduser(begin)

        self.is_new = Folder._make_folder(self.path)

    def archive(self, archive_folder, name='bck'):
        if not isinstance(archive_folder, Folder):
            raise TypeError('path_to_archive expected to be of type Folder')

        if self == archive_folder:
            msg = 'archive_folder can not be the same as {}'.format(self)
            raise ValueError(msg)

        if self.is_parent(archive_folder):
            msg = 'archive_folder can not be a subfolder of {}'.format(self)
            raise ValueError(msg)

        if not self.get_files():
            return

        zip_file_name = os.path.join(
            archive_folder.path,
            datetime.now().strftime('%Y%m%d_%H%M%S_{}.zip'.format(name))
        )
        zipf = zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED)
        files = self.get_files()
        for idx, file in enumerate(files):
            sys.stdout.write(
                'Archiving existing dataset: {:.0%} done\n'.format(
                    (idx+1)/len(files)
                )
            )
            zipf.write(file)
            os.remove(file)
        zipf.close()
        for folder in self.get_folders():
            os.rmdir(folder.path)
        os.rmdir(self.path)
        self.is_new = True
        print('\n"{}" archived to "{}"'.format(self, zip_file_name))

    def is_parent(self, other):
        return self.path == other.path[:len(self.path)]

    def is_empty(self):
        if os.listdir(self.path):
            return False
        return True

    def get_files(self, regex=None):
        if regex:
            re.compile(regex)
            return [
                os.path.join(root, file)
                for root, _, files in os.walk(self.path)
                for file in files
                if re.match(regex, file)
            ]
        else:
            return [
                os.path.join(root, file)
                for root, _, files in os.walk(self.path)
                for file in files
            ]

    def get_folders(self):
        return sorted([
                Folder(root, folder)
                for root, folders, _ in os.walk(self.path)
                for folder in folders
            ],
            reverse=True
        )

    def get_file_name(self, file_name):
        return os.path.join(self.path, str(file_name))

    def file_exists(self, file_name):
        _, file_name = os.path.split(str(file_name))
        print('hiho {}'.format(self.get_file_name(file_name))
        if os.path.isfile(self.get_file_name(file_name)):
            return True
        return False

    @staticmethod
    def _make_folder(path):
        path = os.path.expanduser(str(path))
        if not os.path.exists(path):
            os.makedirs(path)
            return True
        return False

    def __str__(self):
        return self.path

    def __eq__(self, other):
        return self.path == other.path

    def __lt__(self, other):
        return self.path < other.path

    def __gt__(self, other):
        return self.path > other.path


if __name__ == '__main__':
    fo = Folder('../../storage/datasets/wiki/en')
    print(fo)
    fo = Folder('../../storage/datasets/wiki/en', 'raw_data')
    print(fo)
    print(len(fo.get_files()))
    print(len(fo.get_files('.*[0-9]+')))
    print(len(fo.get_files('.*10')))
    print(fo.file_exists(fo.get_files()[0]))
    print(fo.file_exists('blibli'))
