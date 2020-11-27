from trvo_utils.timer import timeit

from with_image_aligning.digits_extractors.clustering_digits_extractor import ClusteringDigitsExtractor, DigitAtPoint
from with_image_aligning.digits_extractors.tests.test_utils import loadDetections, showDigits


def main():
    detections, numOfObservations = loadDetections(1)
    extractor = ClusteringDigitsExtractor()
    digitsAtPoints = extractor.extract(detections, numOfObservations)

    print(len(detections))
    for _ in range(5):
        with timeit():
            digitsAtPoints = extractor.extract(detections, numOfObservations)
    # showDigits(digitsAtPoints)


main()
