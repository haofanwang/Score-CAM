# pip install importlib_resources

import torch
import torch.nn.functional as F
import torchvision.models as models
import os
from os import listdir
from os.path import isfile, join

from utils import *
from cam.socrecam_batch import *

images_path = "images"
images_file = [f for f in listdir(images_path) if isfile(join(images_path, f))]
result_path = "result"

if not os.path.isdir(result_path):
  os.mkdir(result_path)

# alexnet
alexnet = models.alexnet(pretrained=True).eval()
alexnet_model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_10',input_size=(224, 224))
alexnet_scorecam = ScoreCAM(alexnet_model_dict)

input_image_list = list()
for image in images_file:
  input_image = load_image(join(images_path, image))
  input_ = apply_transforms(input_image)
  if torch.cuda.is_available():
    input_ = input_.cuda()
  input_image_list.append(input_)

input_ = torch.cat(input_image_list)
predicted_class = alexnet(input_).max(1)[-1]

scorecam_map = alexnet_scorecam(input_)

# For visualizaiton, clip the input in the range(0, 1)
input_ = torch.where(input_>0, input_, torch.zeros_like(input_))
input_ = torch.where(input_<1, input_, torch.ones_like(input_))

for idx, image in enumerate(images_file):
  store_file_name = join(result_path, "alexnet_"+image.split(".")[0]+".png")
  basic_visualize(input_[idx].cpu(), scorecam_map[idx].type(torch.FloatTensor).cpu(),save_path=store_file_name)

# vgg
vgg = models.vgg16(pretrained=True).eval()
vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name='features_29',input_size=(224, 224))
vgg_scorecam = ScoreCAM(vgg_model_dict)

input_image_list = list()
for image in images_file:
  input_image = load_image(join(images_path, image))
  input_ = apply_transforms(input_image)
  if torch.cuda.is_available():
    input_ = input_.cuda()
  input_image_list.append(input_)

input_ = torch.cat(input_image_list)
predicted_class = vgg(input_).max(1)[-1]

scorecam_map = vgg_scorecam(input_)

# For visualizaiton, clip the input in the range(0, 1)
input_ = torch.where(input_>0, input_, torch.zeros_like(input_))
input_ = torch.where(input_<1, input_, torch.ones_like(input_))

for idx, image in enumerate(images_file):
  store_file_name = join(result_path, "vgg_"+image.split(".")[0]+".png")
  basic_visualize(input_[idx].cpu(), scorecam_map[idx].type(torch.FloatTensor).cpu(),save_path=store_file_name)

# resnet
resnet = models.resnet18(pretrained=True).eval()
resnet_model_dict = dict(type='resnet18', arch=resnet, layer_name='layer4',input_size=(224, 224))
resnet_scorecam = ScoreCAM(resnet_model_dict)

input_image_list = list()
for image in images_file:
  input_image = load_image(join(images_path, image))
  input_ = apply_transforms(input_image)
  if torch.cuda.is_available():
    input_ = input_.cuda()
  input_image_list.append(input_)

input_ = torch.cat(input_image_list)
predicted_class = resnet(input_).max(1)[-1]

scorecam_map = resnet_scorecam(input_)

# For visualizaiton, clip the input in the range(0, 1)
input_ = torch.where(input_>0, input_, torch.zeros_like(input_))
input_ = torch.where(input_<1, input_, torch.ones_like(input_))

for idx, image in enumerate(images_file):
  store_file_name = join(result_path, "resnet_"+image.split(".")[0]+".png")
  basic_visualize(input_[idx].cpu(), scorecam_map[idx].type(torch.FloatTensor).cpu(),save_path=store_file_name)