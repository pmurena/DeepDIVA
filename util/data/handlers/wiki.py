import re
import sys
import bz2
import collections
import pickle

from util.data.handlers.helpers import Folder, Downloader
from util.externals.wikiextractor import WikiExtractor


class GetTheWiki:

    WHICH_WIKI = {
        'en': 'enwiki',
        'fr': 'frwiki',
        'de': 'dewiki',
        'it': 'itwiki'
    }
    URL_DOMAIN = 'dumps.wikimedia.org'
    URL_PATH = '{}/latest'
    URL_FILE = '{}-latest-pages-articles.xml.bz2'

    def __init__(self, output_folder, language='en'):
        self.language = language
        self.root = Folder(output_folder, 'wiki')
        self.path = Folder(self.root, self.language)
        self.train = Folder(self.path, 'train')
        self.val = Folder(self.path, 'val')
        self.test = Folder(self.path, 'test')
        self.data = Folder(self.path, 'raw_data')
        self.bck = Folder(self.root, Folder('archive', language))
        self.downloader = Downloader(
            self.URL_DOMAIN,
            self.URL_PATH.format(self.WHICH_WIKI[self.language]),
            self.URL_FILE.format(self.WHICH_WIKI[self.language])
        )

    def archive(self):
        self.path.archive(self.bck, 'wiki_{}_bck'.format(self.language))
        Folder._make_folder(self.path)
        Folder._make_folder(self.data)

    def extract(self, file_name):
        with open(file_name, 'wb') as dump_xml:
            with bz2.BZ2File(self.get_wiki_zip(), 'rb') as downloaded:
                msg = 'Extracting {} to {}'
                print(msg.format(self.downloader, self.data))
                for data in iter(lambda: downloaded.read(100 * 1024), b''):
                    dump_xml.write(data)
        print('{} extracted to {}'.format(self.downloader, self.data))

    def inflate(self):
        file_name = self.get_wiki_xml()
        print('Inflating {} to {}'.format(file_name, self.data))
        tmp_argv = list(sys.argv)
        sys.argv = list()
        sys.argv.extend(['WikiExtractor.py'])
        sys.argv.extend([file_name])
        sys.argv.extend(['-q'])
        sys.argv.extend(['-o', self.data.path])
        sys.argv.extend(['--processes', '5'])
        WikiExtractor.main()
        sys.argv = list(tmp_argv)
        print('Inflated to {} as {}'.format(
                self.data,
                self.get_wiki_xml()
            )
        )

    def build_corpus_and_vocabulary(self):
        # Build corpus and vocabulary.
        files = self.get_wiki_dump()
        vocabulary = collections.Counter()
        train_corpus = list()
        val_corpus = list()
        test_corpus = list()
        progress_msg = 'working on {} - {:4d}/{} files done\n'
        for idx, file in enumerate(files):
            progress = int(((idx+1)/len(files))*100)
            sys.stdout.write(progress_msg.format(file, idx+1, len(files)))
            with open(file, 'r') as input_f:
                f = re.sub(r'<.*?>', '', input_f.read())
                f = re.sub(
                    r'(\W+)',
                    ' \1 ',
                    f
                )
                f = ''.join([
                    line
                    for line in input_f.readlines()
                    if len(line.split()) > 3
                ])

            file_num = int(file[len(file)-2:])
            if file_num in range(31, 100):
                train_corpus.extend(f)
            elif file_num in range(11, 30):
                val_corpus.extend(f)
            else:
                test_corpus.extend(f)

            vocabulary += collections.Counter(f)

            if progress >= 10 and progress % 10 == 0:
                file_name = 'corpus_{:02d}.pickle'.format(int(progress/10))

                with open(self.train.get_file_name(file_name), 'wb') as c:
                    pickle.dump(train_corpus, c)

                with open(self.val.get_file_name(file_name), 'wb') as c:
                    pickle.dump(val_corpus, c)

                with open(self.test.get_file_name(file_name), 'wb') as c:
                    pickle.dump(test_corpus, c)
                train_corpus.clear()
                val_corpus.clear()
                test_corpus.clear()

        print('corpus saved to {}'.format(self.path))
        voc_file = self.data.get_file_name('vocabulary.pickle')
        with open(voc_file, 'wb') as voc:
            pickle.dump(vocabulary, voc)
        print('Vocabulary saved to {}'.format(voc_file))

    def get(self, new=False):
        if new:
            self.archive()
        self.build_corpus_and_vocabulary()

    def get_wiki_zip(self):
        file_name = self.data.get_file_name(self.downloader)
        if not self.data.file_exists(self.downloader):
            self.downloader.download(file_name)
        return file_name

    def get_wiki_xml(self):
        file_name = self.get_wiki_zip()[:-4]
        if not self.data.file_exists(file_name):
            self.extract(file_name)
        return file_name

    def get_wiki_dump(self):
        regex = 'wiki_[0-9]{2}'
        files = self.data.get_files(regex)
        if not files:
            self.inflate()
            files = self.data.get_files(regex)
        return files

    def __str__(self):
        return str(self.path)


def wiki_mutlitask_hook(getTheWiki_instance):
    if not isinstance(getTheWiki_instance, GetTheWiki):
        raise TypeError('Expecting oject of type GetTheWiki')
    getTheWiki_instance.get()


if __name__ == '__main__':
    wiki = GetTheWiki('data')
    print(wiki)