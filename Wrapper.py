import time
import cv2 as cv
from DetectionDS import TrackShipsUsingDS
from DetectionYOLO import TrackShipsUsingYOLO
from Segmentation import SegmentSea

class WrapperClass:
    def __init__(self, det_path, seg_path, use='both'):
        self.det_path = det_path
        self.seg_path = seg_path
        self.use = use
        self.detection_model = TrackShipsUsingDS(self.det_path)
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
    VIDEO = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_24.mp4"
    cap = cv.VideoCapture(VIDEO)
    det_weights = 'yolov8l.pt'
    seg_weights = "wasr_rn101.pth"
    model = WrapperClass(det_weights, seg_weights, 'segment')
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bboxes, mask = model.infer(frame)
        end = time.time()
        print(f'FPS: {1.0 / (end - start)}')
        start = time.time()
        if model.use == 'detect' or model.use == 'both':
            for i, xmin, ymin, xmax, ymax, p, c in bboxes:
                if c == 'boat':
                    frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    frame = cv.putText(frame, f'ID: {i}, {c}: {p}', (xmax, ymax - 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv.imshow('Detection', frame)
        if model.use == 'segment' or model.use == 'both':
            cv.imshow('Segmentation', mask)
        cv.waitKey(1)
    cap.release()
