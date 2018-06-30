import re
import sys
import bz2
import collections
import pickle

import multiprocessing
from util.data.handlers.helpers import Folder, Downloader
from util.externals.wikiextractor import WikiExtractor

import time


class GetTheWiki:

    VOC_Q = multiprocessing.JoinableQueue()

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
        output_folder = Folder(output_folder, 'wiki')
        self.path = Folder(output_folder, self.language)
        self.data = Folder(self.path, 'raw_data')
        self.bck = Folder(output_folder, Folder('archive', language))
        self.vocabulary = collections.Counter()
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
        msg = 'Extracting {} to {}'
        print(msg.format(self.downloader, self.data))
        with open(file_name, 'wb') as dump_xml:
            with bz2.BZ2File(self.get_wiki_zip(), 'rb') as downloaded:
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
        sys.argv.extend(['--processes', '15'])
        WikiExtractor.main()
        sys.argv = list(tmp_argv)
        print('Inflated to {} as {}'.format(
                self.data,
                self.get_wiki_xml()
            )
        )

    def get_corpus(self, file):
        # print('working on {}'.format(file))
        with open(file, 'r') as input_f:
            f = re.sub(r'<.*?>', '', input_f.read())
            f = re.sub(r'(\W+)', r' \1 ', f)
            f = re.sub(r' {2,}', ' ', f)
            f = re.sub(r'\n{2,}', '\n', f)
            f = f.split('\n')
        r_val = [l for l in f if len(l.split()) > 10]
        GetTheWiki.VOC_Q.put(collections.Counter(' '.join(r_val).split()))
        return r_val

    def add_counter_to_voc(self):
        r_voc = collections.Counter()
        while True:
            try:
                q_val = GetTheWiki.VOC_Q.get()
            except multiprocessing.queues.Empty:
                continue
            if q_val == 'done':
                break
            else:
                print('current queue size estimate {}'.format(
                        GetTheWiki.VOC_Q.qsize()
                    )
                )
                r_voc += q_val
                GetTheWiki.VOC_Q.task_done()
        return r_voc

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


if __name__ == '__main__':
    wiki = GetTheWiki('/home/pat/storage/datasets/')
    print(wiki.get_wiki_dump())
