from torch.utils.data import DataLoader, Dataset
import glob
import os
from torchvision import transforms
import json
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from collections import defaultdict
import torch
import numpy as np
import cv2
from google.colab.patches import cv2_imshow 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision 

transform2 = transforms.Compose([
    # transforms.Resize((224, 224)),  # resize the images to 224x224
    transforms.ToTensor(),         # convert the images to PyTorch tensors
])
## Dataset class --> returns [input, labels]
class BDD_Dataset(Dataset):

  def __init__(self, image_dir, label_dir, transform_=None):
    self.image_dir = image_dir
    self.label_dir = label_dir 
    self.transform_ = transform2
    ## lisr of image paths
    self.image_paths = os.listdir(os.path.join(image_dir))
    self.labels = self._read_labels(self.label_dir)
  
  
  def __getitem__(self, idx):
        """
        Get a single image / label pair.
        """

        label = self.labels[idx]
        img_file = os.path.join(self.image_dir, label['name'])
        img = Image.open(img_file)
        
        img = self.transform_(img)
        target = []
        for box, label in zip(label['boxes'], label['labels']):
          target.append( {'boxes': torch.tensor(box), 'labels': torch.tensor(label)} )
        
        
        return img_file, img, target

  def __len__(self):
    """
        Return length of the dataset
     """
    return len(self.labels)
  

  def _read_labels(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        labels = []
        for item in data:
            name = item['name']
            if not os.path.exists(os.path.join(self.image_dir, name)):
              continue
            attributes = item['attributes']
            labels_data = item['labels']
            labels_info = []
            target = []
            categories = []
            bboxes = []
            for label_data in labels_data:
                category = label_data['category']
                if category == "traffic sign" or category == "traffic light":
                    if 'box2d' in label_data.keys():
                      box2d = list(label_data['box2d'].values())
                      id_ = label_data['id']
                      categories.append(0 if category == 'traffic sign' else 1)
                      bboxes.append(box2d)
                else:
                  continue 
            if len(bboxes) > 0:
              labels.append({'name': name, 'boxes': bboxes, 'labels': categories})
        return labels

  def custom_collate(batch):
    return tuple(zip(*batch))
  

train_dataset = BDD_Dataset(image_dir=train_image_dir, label_dir=train_label_dir, transform_=transform2)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

val_dataset = BDD_Dataset(image_dir=val_image_dir, label_dir=val_label_dir, transform_=transform2)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

if torch.cuda.is_available():
  model = model.cuda()

## optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

## number of epochs 
num_epochs = 100

train_losses = []
train_maps = []
val_maps = []
metric = MeanAveragePrecision()
for epochs in range(num_epochs):
   epoch_loss = 0
   total_loss = 0 
   total_correct = 0 
   total = 0
   
   for i, (img_file, img, target) in enumerate(train_dataloader):
   
     if torch.cuda.is_available():
      
       img = torch.unsqueeze(img[0], 0).cuda()
     model.train()
     targets_dict = {}
     targets_dict['boxes'] = torch.stack([t['boxes'] for t in target[0]])
     targets_dict['labels'] = torch.stack([t['labels'] for t in target[0]])
     if torch.cuda.is_available():
       targets_dict['boxes'] = targets_dict['boxes'].cuda()
       targets_dict['labels'] =  targets_dict['labels'].cuda()
     loss_dict = model(torch.unsqueeze(img[0], 0), targets=[targets_dict])
     
     loss = sum(v for v in loss_dict.values())
     total_loss += loss.cpu().detach().numpy()
     total += len(target[0])
  
     epoch_loss += loss.cpu().detach().numpy()
   
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()

     model.eval()
     with torch.no_grad():
      preds = model(img)
      metric.update(preds, [targets_dict])
     
    
   

   loss_ = total_loss/total
   train_maps.append(metric.compute()['map'])
   
   
   print("For epoch: {}, the total loss is: {}".format(epochs, loss_))
   print(f"Training mAP: {train_maps[-1]}")
   train_losses.append(loss_)
   if epochs % 10 == 0:
     val_maps.append(eval('test'))
     print(f"Validation mAP: {val_maps[-1]}")

