import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class TrackShipsUsingDS:
    def __init__(self, best='best.pt', conf=0.4, max_iou_distance=0.5, max_age=60, n_init=5):
        self.best = best
        self.conf = conf
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.tracker = DeepSort(max_age=self.max_age, max_iou_distance=self.max_iou_distance, n_init=self.n_init)
        self.model = YOLO(self.best)

    def infer(self, image):
        detections = self.model.predict(frame)[0]
        classes = detections.names
        detections = detections.boxes
        xyxy = detections.xyxy.cpu().numpy().tolist()
        cls = detections.cls.cpu().numpy().tolist()
        conf = detections.conf.cpu().numpy().tolist()
        results = []
        for (xmin, ymin, xmax, ymax), p, c in zip(xyxy, conf, cls):
            if p > self.conf:
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], p, c])
        tracks = self.tracker.update_tracks(results, frame=frame)
        results = []
        for track in tracks:
            if track.is_confirmed():
                i = track.track_id
                p = track.get_det_conf()
                c = track.get_det_class()
                xmin, ymin, xmax, ymax = track.to_ltrb(orig=True)
                if p is not None:
                    results.append([i, int(xmin), int(ymin), int(xmax), int(ymax), round(p * 100, 2), classes[c]])
                else:
                    results.append([i, int(xmin), int(ymin), int(xmax), int(ymax), 0.0, classes[c]])
        return results

if __name__ == '__main__':
    VIDEO = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_24.mp4"
    cap = cv.VideoCapture(VIDEO)
    WEIGHTS = 'yolov8l.pt'
    model = TrackShipsUsingDS(WEIGHTS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.infer(frame)
        for i, xmin, ymin, xmax, ymax, p, c in results:
            if c == 'boat':
                frame = cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                frame = cv.putText(frame, f'ID: {i}, {c}: {p}', (xmax, ymax - 10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        cv.imshow('Frame', frame)
        cv.waitKey(1)
cap.release()
