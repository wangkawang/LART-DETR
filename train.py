import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR(r'LART-DETR.yaml')
    # model.load(r'') # loading pretrain weights
    model.train(data=r'data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4,
                device='0',
                project='runs/train',
                )