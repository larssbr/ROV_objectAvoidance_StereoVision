from skimage import feature

class HOG:
    def __init__(self, orientations = 9, pixelsPerCell = (8, 8),
        cellsPerBlock = (3, 3), normalize = False):
        self.orienations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.normalize = normalize

    def describe(self, image):
        hist = feature.hog(image,
        orientations = self.orienations,
        pixels_per_cell = self.pixelsPerCell,
        cells_per_block = self.cellsPerBlock,
        normalise = self.normalize)

        return hist