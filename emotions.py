import tempfile
import doctest
import weakref
import zipfile
import tarfile
import shutil
import glob
import os
import io

from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import requests

from basedir import DATA_FOLDER


RANDOM_STATE = 1


def dump(X, y, prefix='fold', output_dir='.', folds=None, stratified=True,
         **fold_kwargs):
    """Dumps dataframe into file.

    Parameters
    ----------
    folds: int, or None, default: None
        If integer value provided, then data will be saved into this number
        of files each one with equal number of records (except maybe the last
        chunk).

    stratified: bool, default: False
        If True, then `sklearn.StratifiedKFold` is used to split dataset into
        chunks, otherwise - `sklearn.KFold`.

    """
    if folds:
        fold_factory = [KFold, StratifiedKFold][stratified]
        try:
            n_splits = int(folds)
        except ValueError:
            n_splits = 3
        fold_kwargs['n_splits'] = n_splits
        fold = fold_factory(**fold_kwargs)

        template = '%s_{0}_{1:0%d}' % (prefix, len(str(abs(n_splits))))
        if output_dir:
            template = os.path.join(output_dir, template)

        splits = []
        for i, (_, subset) in enumerate(fold.split(X, y)):
            X_subset, y_subset = X[subset], y[subset]
            np.save(template.format('X', i), X_subset)
            np.save(template.format('y', i), y_subset)
            splits.append((i, subset))

        meta = os.path.join(os.path.dirname(template), 'split.txt')
        with open(meta, 'w') as fp:
            lines = ['%d::%s\n' % (i, ','.join(map(str, fold)))
                     for i, fold in splits]
            fp.writelines(lines)

    else:
        np.save(prefix + '_X', X)
        np.save(prefix + '_y', y)


def load(data_folder, prefix, subset_index=None):
    """Loads data chunk saved with `dump()` function.

    Examples
    --------
    >>> X, y = load(DATA_FOLDER, 'emotions', subset_index=1)
    >>> X.shape[0] == y.shape[0]
    True

    """
    assert os.path.exists(data_folder), 'Data folder not found'
    filenames = list(glob.glob1(data_folder, '*.npy'))
    assert len(filenames) > 0, 'No dumped data files found'
    pad = len(filenames[0].split('_')[-1].strip('.npy'))

    if subset_index is not None:
        assert 0 <= subset_index < len(filenames), 'Wrong subset index'
        template = '{0}_{1}_{2:0%d}.npy' % pad
    else:
        template = '{0}_{1}.npy'
    for suffix in ('X', 'y'):
        filename = template.format(prefix, suffix, subset_index)
        filepath = os.path.join(data_folder, filename)
        assert os.path.exists(filepath), 'File not found: %s' % filepath
        yield np.load(filepath)


class Dataset:
    """Utility class to load and prepare (if needed) some dataset.

    It is simpler to keep data in some remote folder and upload it when needed.
    This way allows to deploy classifiers on dedicated host with less
    difficulties.

    Parameters
    ----------
    dataset_folder: str
        A path to unzip or download dataset.
    """

    _registry = weakref.WeakValueDictionary()
    _sep = '::'

    @staticmethod
    def register(name, cls):
        Dataset._registry[name] = cls

    def __new__(cls, name, *args, **kwargs):
        if not issubclass(cls, Dataset):
            return object.__new__(cls)
        target = Dataset._registry.get(name)
        if not target:
            targets = ", ".join(list(sorted(Dataset._registry.keys())))
            raise ValueError(
                "Wrong dataset name: '%s'. Available datasets are %s" %
                (target, targets))
        return object.__new__(target)

    def __init__(self, name, dataset_folder=None, cache=True):
        self.name = name
        self.dataset_folder = os.path.expanduser(dataset_folder)
        self.cache = cache
        self._dataset = None
        self._prepared = None

    @property
    def content(self):
        return self._dataset

    @property
    def prepared(self):
        return self._prepared

    @property
    def cache_log(self):
        return os.path.join(self.dataset_folder, '.cache')

    def load_from_url(self, url, archive=False, name_in_archive=None):
        if self.cache and self._load_from_cache(url):
            return

        response = requests.get(url)
        _, tmp = tempfile.mkstemp()
        with open(tmp, 'wb') as fp:
            fp.write(response.content)

        filename = tmp
        cache_name = os.path.basename(filename) + '.cache'
        cache_file = os.path.join(self.dataset_folder, cache_name)
        shutil.copy(filename, cache_file)
        cache_entry = [url]

        if archive:
            name_in_archive = name_in_archive or self.name + '.csv'
            cache_entry.extend(['archive', cache_file, name_in_archive])
            self.load_from_archive(filename, name_in_archive)
        else:
            cache_entry.extend(['plain', cache_file])
            self.load_from_file(filename)

        self._cache_log(self._sep.join(cache_entry))

    def load_from_archive(self, archive_name, file_in_archive):
        _, ext = os.path.splitext(archive_name)

        if zipfile.is_zipfile(archive_name):
            with zipfile.ZipFile(archive_name) as arch:
                content = arch.read(file_in_archive)

        elif tarfile.is_tarfile(archive_name):
            with tarfile.open(archive_name) as arch:
                fp = arch.extractfile(file_in_archive)
                content = fp.read()

        else:
            raise ValueError('Cannot recognize archive format')

        buf = io.StringIO(content.decode(encoding='utf-8'))
        buf.seek(0)
        self._dataset = pd.read_csv(buf)

    def load_from_file(self, filename):
        readers = [
            ('csv', pd.read_csv),
            ('dat', pd.read_pickle),
            ('hdf5', lambda path: pd.read_hdf(path, key=self.name))
        ]
        _, ext = os.path.splitext(filename)

        if ext:
            try:
                reader = dict(readers)[ext.strip('.')]
            except LookupError:
                pass
            else:
                self._dataset = reader(filename)
                return

        for _, reader in readers:
            try:
                self._dataset = reader(filename)
                return
            except:
                pass

        raise ValueError("File format wasn't recognized. Cannot load dataset")

    def _load_from_cache(self, url):
        if not os.path.exists(self.cache_log):
            return False

        for line in open(self.cache_log):
            cached_url, filetype, *rest = line.strip().split(self._sep)

            if url == cached_url:
                try:
                    if filetype == 'archive':
                        archive_name, file_in_archive = rest
                        self.load_from_archive(archive_name, file_in_archive)
                    elif filetype == 'plain':
                        [filename] = rest
                        self.load_from_file(filename)
                except Exception as e:
                    print('Cannot load cache: %s' % e)
                    return False
                else:
                    return True

        return False

    def _cache_log(self, record):
        with open(self.cache_log, 'a') as fp:
            fp.write(record)
            fp.write('\n')

    def prepare(self):
        """Prepares dataset before it can be used.

        Default implementation does nothing.
        """
        pass


class FER2013Dataset(Dataset):

    VERBOSE_EMOTION = [
        'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    IMAGE_SIDE = 48
    IMAGE_SIZE = IMAGE_SIDE ** 2

    def __init__(self, name, dataset_folder='.', cache=True):
        super(FER2013Dataset, self).__init__(name, dataset_folder, cache)

    @classmethod
    def get_name(cls):
        return 'fer2013'

    def prepare(self):

        def to_pixels(row):
            return np.array([float(px) for px in row.split()])

        emotions_data = self._dataset
        labels = emotions_data['emotion'].apply(int)
        labels.name = 'label'
        verbose = labels.apply(
            lambda emotion_label: self.VERBOSE_EMOTION[emotion_label])
        verbose.name = 'verbose'
        usage = emotions_data.Usage.map({
            'Training': 'train',
            'PublicTest': 'valid',
            'PrivateTest': 'test'
        })
        usage.name = 'subset'

        flat_images = pd.DataFrame(
            np.r_[[to_pixels(row) for row in emotions_data.pixels.values]],
            columns=['p' + str(i) for i in range(1, self.IMAGE_SIZE + 1)])

        processed_data = pd.DataFrame(
            pd.concat([labels, verbose, usage, flat_images], axis=1))
        self._prepared = processed_data

    @staticmethod
    def label_to_verbose(label):
        try:
            return FER2013Dataset.VERBOSE_EMOTION[label]
        except (LookupError, TypeError):
            raise ValueError("Wrong emotion label: '%d'" % label)


Dataset.register('fer2013', FER2013Dataset)


if __name__ == '__main__':
    doctest.testmod()
