import time
import cv2 as cv
from DetectionDS import TrackShipsUsingDS
from DetectionYOLO import TrackShipsUsingYOLO
from PredictMotionLRClass import MotionPredictor
from Segmentation import SegmentSea

class WrapperClass:
    def __init__(self, use='both', seg_path=None, det_path=None, det_model='yolo', prediction=True):
        self.use = use
        if self.use == 'segment':
            self.seg_path = seg_path
            self.segmentation_model = SegmentSea(self.seg_path)
        elif self.use == 'detect':
            self.det_path = det_path
            self.det_model = det_model
            self.prediction = prediction
            if self.det_model == 'yolo':
                self.detection_model = TrackShipsUsingYOLO(self.det_path)
            else:
                self.detection_model = TrackShipsUsingDS(self.det_path)
            if self.prediction:
                self.motion_predictor = MotionPredictor()
        else:
            self.seg_path = seg_path
            self.segmentation_model = SegmentSea(self.seg_path)
            self.det_path = det_path
            self.det_model = det_model
            self.prediction = prediction
            if self.det_model == 'yolo':
                self.detection_model = TrackShipsUsingYOLO(self.det_path)
            else:
                self.detection_model = TrackShipsUsingDS(self.det_path)
            if self.prediction:
                self.motion_predictor = MotionPredictor()

    def infer(self, image, data=None):
        if self.use == 'segment':
            mask = self.segmentation_model.infer(image)
            return None, mask, None
        elif self.use == 'detect':
            bboxes = self.detection_model.infer(image)
            for xmin, ymin, xmax, ymax, i, c in bboxes:
                x, y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                self.motion_predictor.update(i, x, y)
            self.motion_predictor.update()
            preds = self.motion_predictor.predict()
            return bboxes, None, preds
        else:
            bboxes = self.detection_model.infer(image)
            for xmin, ymin, xmax, ymax, i, c in bboxes:
                x, y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
                self.motion_predictor.update(i, x, y)
            self.motion_predictor.update()
            preds = self.motion_predictor.predict()
            mask = self.segmentation_model.infer(image)
            return bboxes, mask, preds

if __name__ == '__main__':
    VIDEO = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_1.mp4"
    cap = cv.VideoCapture(VIDEO)
    det_weights = 'yolov8l.pt'
    seg_weights = "wasr_rn101.pth"
    model = WrapperClass('both', seg_weights, det_weights, 'yolo', True)
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bboxes, mask, preds = model.infer(frame)
        end = time.time()
        print(f'FPS: {1.0 / (end - start)}')
        start = time.time()
        out = frame.copy()
        if model.use == 'detect' or model.use == 'both':
            for xmin, ymin, xmax, ymax, i, c in bboxes:
                out = cv.rectangle(out, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                out = cv.putText(out, f'ID: {i}, {c}', (xmin, ymin + 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            for i in preds:
                for x, y in preds[i]:
                    if x is not None and y is not None:
                        out = cv.circle(out, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv.imshow('Detection', out)
        if model.use == 'segment' or model.use == 'both':
            out = cv.addWeighted(mask, 0.3, out, 0.7, 0.0)
            cv.imshow('Segmentation', out)
        cv.waitKey(1)
    cap.release()
