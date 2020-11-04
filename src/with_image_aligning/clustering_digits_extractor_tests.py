from with_image_aligning.clustering_digits_extractor import ClusteringDigitsExtractor


def loadDetections():
    from pickle import load
    with open("digit_detections_1.pcl", "rb") as f:
        numOfObservations = 386
        return load(f), numOfObservations


def main():
    detections, numOfObservations = loadDetections()
    extractor = ClusteringDigitsExtractor()
    digitsAtPoints = extractor.extract(detections, numOfObservations)
    # TODO: sort by point.x
    digitsAtPoints.sort(key=lambda d: d.point[0])
    # TODO: visualize
    for digitAtPoint in digitsAtPoints:
        print(digitAtPoint.digit, digitAtPoint.point)



main()
