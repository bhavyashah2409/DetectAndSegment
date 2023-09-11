from ultralytics import YOLO

class TrackShipsUsingYOLO:
    def __init__(self, best='best.pt', conf=0.0, iou=0.5):
        self.best = best
        self.conf = conf
        self.iou = iou
        self.model = YOLO(self.best)

    def infer(self, image):
        results = model.track(frame, conf=self.conf, iou=self.iou, persist=True)[0]
        classes = results.names
        results = results.boxes
        xyxy = results.xyxy.cpu().numpy().tolist()
        cls = results.cls.cpu().numpy().tolist()
        conf = results.conf.cpu().numpy().tolist()
        ids = results.id
        ids = ids.cpu().numpy().tolist() if ids is not None else [0]
        results = []
        for i, (xmin, ymin, xmax, ymax), p, c in zip(ids, xyxy, conf, cls, ids):
            results.append([i, xmin, ymin, xmax, ymax, p, c])
        return results

if __name__ == '__main__':
    model = DetectShips(r'path\to\best.pt')
    model.infer(image)
