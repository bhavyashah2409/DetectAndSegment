import time
import cv2 as cv
from DetectionDS import TrackShipsUsingDS
from DetectionYOLO import TrackShipsUsingYOLO
from Segmentation import SegmentSea

class WrapperClass:
    def __init__(self, det_path, seg_path, det_model='yolo', use='both'):
        self.det_path = det_path
        self.seg_path = seg_path
        self.use = use
        if self.det_model == 'ds':
            self.detection_model = TrackShipsUsingDS(self.det_path)
        else:
            self.detection_model = TrackShipsUsingYOLO(self.det_path)
        self.segmentation_model = SegmentSea(self.seg_path)

    def infer(self, image):
        if self.use == 'detect':
            bboxes = self.detection_model.infer(image)
            return bboxes, None
        elif self.use == 'segment':
            mask = self.segmentation_model.infer(image)
            return None, mask
        bboxes = self.detection_model.infer(image)
        mask = self.segmentation_model.infer(image)
        return bboxes, mask

if __name__ == '__main__':
    VIDEO = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_1.mp4"
    cap = cv.VideoCapture(VIDEO)
    det_weights = 'yolov8l.pt'
    seg_weights = "wasr_rn101.pth"
    model = WrapperClass(det_weights, seg_weights, 'yolo', 'detect')
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bboxes, mask = model.infer(frame)
        end = time.time()
        print(f'FPS: {1.0 / (end - start)}')
        start = time.time()
        out = frame.copy()
        if model.use == 'detect' or model.use == 'both':
            for xmin, ymin, xmax, ymax, i, c in bboxes:
                if c == 'boat':
                    out = cv.rectangle(out, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    out = cv.putText(out, f'ID: {i}, {c}', (xmin, ymin + 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv.imshow('Detection', out)
        if model.use == 'segment' or model.use == 'both':
            out = cv.addWeighted(mask, 0.3, out, 0.7, 0.0)
            cv.imshow('Segmentation', out)
        cv.waitKey(1)
    cap.release()
