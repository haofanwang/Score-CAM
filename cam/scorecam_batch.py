import torch
import torch.nn.functional as F
from cam.basecam import *


class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        # predication on raw input
        logit = self.model_arch(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
        else:
            predicted_class = class_idx.long()  # assume the class_idx in tensor form

        logit = F.softmax(logit, dim=1)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            logit = logit.cuda()


        predicted_class = predicted_class.reshape(-1, 1)

        activations = self.activations['value']
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((b, 1, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            for i in range(k):

                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(saliency_map, size=(
                    h, w), mode='bilinear', align_corners=False)


                # normalize to 0-1
                saliency_max = saliency_map.view(b, -1).max(dim=1)[0]
                saliency_max = saliency_max.reshape(
                    b, 1, 1, 1).repeat(1, 1, h, w)
                saliency_min = saliency_map.view(b, -1).min(dim=1)[0]
                saliency_min = saliency_min.reshape(
                    b, 1, 1, 1).repeat(1, 1, h, w)
                norm_saliency_map = (saliency_map - saliency_min) / \
                    (saliency_max - saliency_min + 1e-7)

                # how much increase if keeping the highlighted region
                # predication on masked input
                output = self.model_arch(input * norm_saliency_map)
                output = F.softmax(output, dim=-1)
                score = output[torch.arange(predicted_class.size(0)).unsqueeze(1), predicted_class]
                # Apply the torch.where function, so the score of saliency_map.max() == saliency_map.min() instance is 0.
                score = torch.where(saliency_map.view(b, -1).max(dim=1)[0].reshape(b, 1) > saliency_map.view(b, -1).min(dim=1)[0].reshape(b, 1),
                                    score, torch.zeros_like(score))

                score = score.reshape(b, 1, 1, 1).repeat(1, 1, h, w)
                score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min = score_saliency_map.view(b, -1).min(dim=1)[0]
        score_saliency_map_min = score_saliency_map_min.reshape(
            b, 1, 1, 1).repeat(1, 1, h, w)
        score_saliency_map_max = score_saliency_map.view(b, -1).max(dim=1)[0]
        score_saliency_map_max = score_saliency_map_max.reshape(
            b, 1, 1, 1).repeat(1, 1, h, w)

        # count_nonzero is only available after pytorch 1.7.0
        if len(((score_saliency_map_max - score_saliency_map_min) == 0).nonzero(as_tuple=False)) != 0:
            raise Exception

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(
            score_saliency_map_max - score_saliency_map_min).data
        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
