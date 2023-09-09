import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class TrackShips:
    def __init__(self, best_model, max_age, min_conf):
        self.best_model = best_model
        self.max_age = max_age
        self.min_conf = min_conf
        self.tracker = DeepSort(max_age=self.max_age)
        self.model = YOLO(self.best_model)

    def infer(self, image):
        detections = model.predict(frame)[0]
        results = []
        for xmin, ymin, xmax, ymax, p, c in detections.boxes.data.cpu().numpy().tolist():
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], p, c])
        tracks = tracker.update_tracks(results, frame=frame)
        results = []
        for track in tracks:
            if track.is_confirmed():
                i = track.track_id
                p = track.get_det_conf()
                c = track.get_det_class()
                xmin, ymin, xmax, ymax = track.to_ltrb(orig=True)
                results.append([i, xmin, ymin, xmax, ymax, p, c])
        return results

if __name__ == '__main__':
    model = DetectShips(r'path\to\best.pt')
    model.infer(image)
