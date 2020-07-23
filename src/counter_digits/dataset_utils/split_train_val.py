import math
from pathlib import Path

from PIL import ImageMath
from trvo_utils.annotation import PascalVocXmlParser

from counter_digits.dataset_utils.extract_dataset import list_files


class DigitsDataset:
    imagesExtensions = ['jpg', 'jpeg', 'png']

    def __init__(self, datasetRootDir):
        self._dirs = self._digitDirs(datasetRootDir)

    @staticmethod
    def _digitDirs(rootDir):
        dirs = sorted(Path(rootDir).rglob("digits"))
        assert isinstance(dirs, list)
        for d in dirs: assert d.is_dir()
        return dirs

    def _annotationFiles(self, type):
        for d in self._dirs:
            for annFile in sorted(d.glob(f'*.{type}')):
                yield annFile

    def stats(self):
        stats = {l: 0 for l in range(10)}
        # load annotation files
        for annFile in self._annotationFiles('xml'):
            p = PascalVocXmlParser(annFile)
            for l in p.labels():
                l = int(l)
                stats[l] += 1
        return stats

    def _annotatedImagesPerDirectory(self):
        result = dict()
        for d in self._dirs:
            result[d] = list(list_files([d], self.imagesExtensions))
        return result

    @staticmethod
    def _split_items(splitRatio, items):
        assert 0 <= splitRatio <= 1
        split1_len = math.ceil(len(items) * splitRatio)
        split1 = items[:split1_len]
        split2 = items[split1_len:]
        return split1, split2

    def train_val_split(self, splitRatio=.8, saveTo=None):
        trainImages = []
        valImages = []

        # split per directory
        itemsPerDirectory: dict = self._annotatedImagesPerDirectory()
        for d, imageFiles in itemsPerDirectory.items():
            trainItems, valItems = self._split_items(splitRatio, imageFiles)
            trainImages.extend(trainItems)
            valImages.extend(valItems)

        if saveTo is not None:
            trainFileName, valFileName = saveTo
            self.saver.save_split(trainImages, valImages, trainFileName, valFileName)

        return trainImages, valImages

    class saver:
        @classmethod
        def save_split(self, trainItems, valItems, trainFileName, valFileName):
            self._save_items(trainItems, trainFileName)
            self._save_items(valItems, valFileName)

        @staticmethod
        def _save_items(items, file_name):
            lastItemIndex = len(items) - 1
            with open(file_name, "wt") as f:
                for i, item in enumerate(items):
                    f.write(item)
                    if i < lastItemIndex:
                        f.write('\n')


def main_stats():
    rootDir = "/hdd/Datasets/counters"
    ds = DigitsDataset(rootDir)
    stats = ds.stats()
    print(stats)


def main_tran_val_split():
    rootDir = "/hdd/Datasets/counters"
    ds = DigitsDataset(rootDir)

    saveTo = 'train_split.txt', 'val_split.txt'
    trainImages, valImages = ds.train_val_split(splitRatio=.8, saveTo=saveTo)
    print(len(trainImages), len(valImages))


def main_split_test():
    s1, s2 = DigitsDataset._split_items(.5, [1, 2, 3])
    assert s1 == [1, 2] and s2 == [3]

    s1, s2 = DigitsDataset._split_items(.5, [1, 2])
    assert s1 == [1] and s2 == [2]

    s1, s2 = DigitsDataset._split_items(.9, [1, 2, 3])
    assert s1 == [1, 2, 3] and s2 == []

    s1, s2 = DigitsDataset._split_items(1, [1, 2, 3])
    assert s1 == [1, 2, 3] and s2 == []

    s1, s2 = DigitsDataset._split_items(.1, [1, 2, 3])
    assert s1 == [1] and s2 == [2, 3]

    s1, s2 = DigitsDataset._split_items(0, [1, 2, 3])
    assert s1 == [] and s2 == [1, 2, 3]

    s1, s2 = DigitsDataset._split_items(0, [])
    assert s1 == [] and s2 == []

    s1, s2 = DigitsDataset._split_items(.9, [])
    assert s1 == [] and s2 == []


if __name__ == '__main__':
    main_tran_val_split()
