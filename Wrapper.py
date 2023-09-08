from Detection import DetectShips
from Segmentation import SegmentSea

class WrapperClass:
    def __init__(self, det_path, seg_path):
        self.det_path = det_path
        self.seg_path = seg_path
        self.detection_model = DetectShips(self.det_path)
        self.segmentation_model = SegmentSea(self.seg_path)

    def infer(self, image):
        bboxes = self.detection_model.infer(image)
        mask = self.segmentation_model.infer(image)
        return bboxes, mask

    def main(self):
        pass
