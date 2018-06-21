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
        self.data = Folder(self.path, 'row_data')
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
        corpus = list()
        progress_msg = 'working on {} - {:4d}/{} files done\r'
        for idx, file in enumerate(files):
            progress = int(((idx+1)/len(files))*100)
            sys.stdout.write(progress_msg.format(file, idx+1, len(files)))
            with open(file, 'r') as input_f:
                    f = re.sub(
                        r'<.*?>|_|\W+|[0-9]+|[ ]{2,}|\n',
                        ' ',
                        input_f.read()
                    ).lower().split()
            corpus.extend(f)
            vocabulary += collections.Counter(f)
            if progress >= 10 and progress % 10 == 0:
                file_name = self.data.get_file_name(
                    'corpus_{:02d}.pickle'.format(int(progress/10))
                )

                with open(file_name, 'wb') as c:
                    pickle.dump(corpus, c)
                corpus.clear()
        print('corpus saved to {}'.format(self.data))
        voc_file = self.data.get_file_name('vocabulary.pickle')
        with open(voc_file, 'wb') as voc:
            pickle.dump(vocabulary, voc)
        print('Vocabulary saved to {}'.format(voc_file))

    def get(self, new=False):
        if new:
            self.archive()
        return self.build_corpus_and_vocabulary()

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


if __name__ == '__main__':
    wiki = GetTheWiki('data')
    print(wiki)
