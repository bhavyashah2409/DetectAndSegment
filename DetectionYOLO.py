from ultralytics import YOLO

class TrackShipsUsingYOLO:
    def __init__(self, best='best.pt', conf=0.3, iou=0.9):
        self.best = best
        self.conf = conf
        self.iou = iou
        self.model = YOLO(self.best)

    def infer(self, image):
        results = model.track(frame, persist=True, conf=self.conf, iou=self.iou)[0].cpu()
        classes = results.names
        bboxes = results.boxes.xyxy.cpu().numpy().tolist()
        cls = results.boxes.cls.cpu().numpy().tolist()
        conf = results.boxes.conf.cpu().numpy().tolist()
        ids = results.boxes.id
        if ids is not None:
            ids = ids.cpu().numpy().tolist()
        else:
            ids = [0 for _ in bboxes]
        for i, (xmin, ymin, xmax, ymax), p, c in zip(ids, xyxy, conf, cls, ids):
            results.append([i, xmin, ymin, xmax, ymax, p, c])
        return results

if __name__ == '__main__':
    model = DetectShips(r'path\to\best.pt')
    model.infer(image)
