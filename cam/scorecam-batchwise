import torch
import torch.nn.functional as F
from cam.basecam import *


class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM
    """

    def __init__(self, model_dict):
        super().__init__(model_dict)
        
    def get_masked_imgs(self, imgs, activations):
      b, d, r, c = imgs.shape
      _, A, _, _ = activations.shape
      imgs = imgs.reshape(-1)
      imgs = imgs.repeat(A)
      activations = activations.permute(1,0,2,3)
      activations = activations.repeat(1,1,d,1)
      activations = activations.reshape(-1)
      mul = activations*imgs
      mul = mul.reshape(-1,d,r,c)
      return mul

    def activation_wise_normalization(self, activations):
      b,f,h,w = activations.shape
      activations = activations.view(-1,h*w)
      max_ = activations.max(dim=1)[0]
      min_ = activations.min(dim=1)[0]
      check = ~max_.eq(min_)
      max_ = max_[check]
      min_ = min_[check]
      activations = activations[check,:]
      sub_ =  max_ - min_
      sub_1 = activations - min_[:,None]
      norm = sub_1 / sub_[:,None]
      norm = norm.view(b,-1,h,w)
      return norm  

    def get_scores(self, imgs, targets):
      b, _, _, _ = imgs.shape
      total_scores = []
      class MyDataloader(torch.utils.data.Dataset):
        def __init__(self, images):
            self.images = images
        def __len__(self):
            return self.images.shape[0]
        def __getitem__(self, idx):
            return self.images[idx, :, :, :]
            
      train_data = MyDataloader(imgs)
      train_loader = torch.utils.data.DataLoader(train_data,
                                                shuffle=False,
                                                num_workers=0,
                                                batch_size=50)
      for batch_images in train_loader:
        scores = self.sub_forward(batch_images)
        scores = F.softmax(scores, dim=1)
        labels = targets.long()
        scores = scores[:,labels]
        total_scores.append(scores)
      total_scores = torch.cat(total_scores,dim=0)
      total_scores = total_scores.view(-1)
      return total_scores
      
    
    def get_cam(self, activations, scores):
      b,f,h,w = activations.shape
      cam = activations*scores[None,:,None,None]
      cam = cam.sum(1, keepdim=True)
      return cam

    def forward(self, imgs, class_idx=None, retain_graph=False):
        logit = self.model_arch(input).cuda()
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        with torch.no_grad():
            batch_size, D, H, W = imgs.shape
            self.model_arch.zero_grad()
            score.backward(retain_graph=retain_graph)
            activations = self.activations['value']
            
            y = F.relu(activations)
            y = F.interpolate(y, (H, W), mode='bilinear', align_corners=False)
            y = self.activation_wise_normalization(y)
            z = self.get_masked_imgs(imgs, y)
            z = self.get_scores(z, class_idx)
            y = self.get_cam(y,z)
            y = F.relu(y)
            y = normalize_tensor(y)
            y = y.squeeze_(0).detach().clone()
            return y
    
    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
