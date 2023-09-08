class SegmentSea:
    def __init__(self, best_model):
        self.best_model = best_model
        self.model = SegmentationModel(self.best_model)

    def infer(self, image):
        mask = self.model(image)
        return mask
