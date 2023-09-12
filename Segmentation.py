import cv2
import torch
import numpy as np
from PIL import Image
import wasr.models as models

class SegmentSea:
    def __init__(self, best_model, arch='wasr_resnet101'):
        self.best_model = best_model
        self.arch = arch
        self.model = models.get_model(self.arch, pretrained=False)
        self.state_dict = torch.load(best_model, map_location='cpu')
        if 'model' in self.state_dict:
            self.state_dict = self.state_dict['model']
        self.model.load_state_dict(self.state_dict)
        self.model = self.model.eval().cuda()
        
    def infer(self, image):
        H, W, _ = image.shape
        img = torch.from_numpy(image) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = img.permute(2,0,1).unsqueeze(0)
        img = img.float()
        feat = {'image': img.cuda()}
        res = self.model(feat)
        probs = res['out'].detach().softmax(1).cpu()
        probs = torch.nn.functional.interpolate(probs, (H,W), mode='bilinear')
        preds = probs.argmax(1)[0]
        preds_rgb = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)[preds]
        return preds_rgb

if __name__ == '__main__':
    img_path = ""
    model_path = r"wasr\wasr_rn101.pth"
    model = SegmentSea(model_path)
    img = np.array(Image.open(img_path))
    img = model.infer(img)
    img.show()
