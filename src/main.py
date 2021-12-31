# -*- coding: utf-8 -*-
import sys
import os
import glob
import random
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, Subset, ConcatDataset
from torch.nn import CrossEntropyLoss
# from PIL import Image
from tqdm import tqdm

from model import ResNet

"""# 데이터셋 만들기"""

class finalDataset(Dataset):
  def __init__(self, original):

    self.img_data = []
    self.class_data = []
    original_length = len(original)

    for i in range(0, original_length):
      self.img_data.append(original[i][0])
      self.class_data.append(original[i][1])
    
  def __len__(self):
    return len(self.img_data)
  
  def __getitem__(self, idx):
    img_val = self.img_data[idx]
    class_val = self.class_data[idx]
    return img_val, class_val


def resnet18():
    return ResNet()


def trainFunction():
  model = resnet18().to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0001)

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3, verbose=True)

  """# Train"""

  epochs = 100

  # Variables for lr scheduling and early stopping
  best_valid_loss = 9999   
  patience = 0  

  epoch_list = []
  train_acc_list = []
  valid_acc_list = []
  train_batches = len(train_loader)
  valid_batches = len(val_loader)

  # Commented out IPython magic to ensure Python compatibility.
  for epoch in range(0, epochs):

      train_loss = 0
      train_total = 0
      train_correct = 0
      model.train()

      with tqdm(train_loader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
          tepoch.set_description(f"Epoch {epoch+1}") 

          inputs, labels = inputs.to(device), labels.to(device)
          optimizer.zero_grad()

          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          train_loss += loss.item()

          _, predicted = outputs.max(1)
          train_total += labels.size(0)
          train_correct += predicted.eq(labels).sum().item()

      train_acc = train_correct / train_total
      train_loss = train_loss / train_batches

      valid_loss = 0
      valid_total = 0
      valid_correct = 0
      model.eval()

      with torch.no_grad():
          for inputs, labels in val_loader:
              inputs, labels = inputs.to(device), labels.to(device)
              outputs = model(inputs)
              loss = F.cross_entropy(outputs, labels)
              valid_loss += loss.item()
              _, predicted = torch.max(outputs.data, 1)
              valid_total += labels.size(0)
              valid_correct += predicted.eq(labels).sum().item()
      valid_acc = valid_correct / valid_total
      valid_loss = valid_loss / valid_batches

      # 그래프 그리기 위해
      train_acc_list.append(train_acc)
      epoch_list.append(epoch)
      valid_acc_list.append(valid_acc)


      # Save best model
      # early stopping 을 위해 만든 것 (overfitting)
      if best_valid_loss > valid_loss:
          torch.save(model.state_dict(), '../model/' + modelFileName) #모델 저장
          best_valid_loss = valid_loss
          patience = 0

      print('[%d/%d] TrainLoss: %.3f, ValLoss: %.3f | TrainAcc: %.2f, ValAcc: %.2f' % (epoch+1, epochs, train_loss, valid_loss, train_acc * 100, valid_acc * 100))
      
      print('\n')

      scheduler.step(metrics=valid_loss)

      # 모델 발전 7번 이상 없을 때
      if patience == 7:
        break
      
      patience += 1

def testFunction():
  """# 테스트

  ### 모델 로드
  """

  # Load best model
  loaded = resnet18().to(device)
  loaded.load_state_dict(torch.load('../model/' + modelFileName))
  criterion = nn.CrossEntropyLoss()


  """### 테스트 코드"""

  # Test
  loaded.eval()

  test_loss = 0
  test_correct = 0
  idx = 0
  test_total = 0

  with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
      x = x.to(device)
      y = y.to(device)
      outputs = loaded(x)
      loss = criterion(outputs, y)
      
      test_loss += loss.item()
      _, predicted = outputs.max(1)
      test_correct += predicted.eq(y).sum().item()
      test_total += y.size(0)
      
      
      if i == 0:
          test_preds = predicted
          class_true = y
      else:
          test_preds = torch.cat((test_preds, predicted), dim=0)
          class_true = torch.cat((class_true, y), dim = 0)

      idx += 1
              
  test_preds = test_preds.cpu()

  print('TEST loss: %.3f, acc: %.2f' % (test_loss/len(test_loader), (test_correct/test_total) * 100))



# if __name__ == "main" :

# print("hello")

traintest = sys.argv[1]
dataFileName = sys.argv[2]
modelFileName = sys.argv[3]


# 데이터셋 로드
trainset = torch.load('../data/' + dataFileName)
# trainset = torch.load('/Users/hwiwon/Google Drive/내 드라이브/인지프/data' + dataFileName)

SEED = 42
tr_idx, valtest_idx = train_test_split(list(range(len(trainset))), test_size=0.2, random_state=SEED, stratify=trainset.class_data)
tr_split = Subset(trainset, tr_idx)
valtest_split = Subset(trainset, valtest_idx)

valtest_split = finalDataset(valtest_split)

val_idx, test_idx = train_test_split(list(range(len(valtest_split))), test_size=0.5, random_state=SEED, stratify=valtest_split.class_data)
val_split = Subset(valtest_split, val_idx)
test_split = Subset(valtest_split, test_idx)

##subset 사용
num_workers = int(cpu_count() / 2)

train_loader = DataLoader(
    dataset=tr_split,
    batch_size=256,
    num_workers=num_workers
)
val_loader = DataLoader(
    dataset=val_split,
    batch_size=256,
    num_workers=num_workers
)
test_loader = DataLoader(
    dataset=test_split,
    batch_size=256,
    num_workers=num_workers
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if traintest == "train" :
  trainFunction()

elif traintest == "test" :
  testFunction()

else :
  print("argument error/ choose train or test")