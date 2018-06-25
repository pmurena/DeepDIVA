import sys
import urllib


class Downloader:

    def __init__(self, domain, directory, file_name='', secure=True):
        self.protocol = 'https' if secure else 'http'
        self.domain = domain.strip('/')
        self.directory = directory.strip('/')
        self.file_name = file_name.strip('/')

    def download(self, target_file_name='./tmp_file_19800223', end_point=''):
        return urllib.request.urlretrieve(
            self.get_url(end_point),
            target_file_name,
            self._download_reporthook
        )

    def _download_reporthook(self, blocknum, blocksize, totalsize):
        """Report hook to print downloading progress of urllib.request

        Defines the report hook used to call urllib.request.urlretrieve(url,
        filename=None, reporthook=None, data=None). This function will be
        called by urlretrieve() after each block is received. Thus, the
        downloading percentage is computed and displayed to the standard output
        for each incoming block.

        Args:
            blocknum (int): Number of received blocks
            blocksize (int): Size of the incoming network blocks
            totalsize (int): Size of the incoming file

        Returns:
            None

        Raises:
            None
        """
        # Compute download progress
        progress = blocknum * blocksize / totalsize

        # Choose between carriage return and new line
        if progress < 100:
            str_end = '\r'
        else:
            str_end = '\n'

        # Write download progress to standard output
        sys.stdout.write('Downloading "{}": {:.2%}'.format(
            self.file_name,
            progress
            )
        )

        return

    def get_url(self, end_point=''):
        end_point = self.file_name if not end_point else end_point
        return '{}://{}/{}/{}'.format(
            self.protocol,
            self.domain,
            self.directory,
            end_point
        )

    def __str__(self):
        return self.file_name
