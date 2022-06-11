import torch
import torch.nn.functional as F

def Scaled_EPE_Loss(pred_flow, target_flow):

    b, _, h, w = pred_flow.size()
    scaled_target = F.interpolate(target_flow, (h,w), mode='area')
    return torch.norm(scaled_target-pred_flow, 2, 1).sum() / b

def MultiScale_EPE_Loss(model_output, target_flow):
    weights = [0.005, 0.01, 0.02, 0.08, 0.32]
    loss = 0.0
    for out, weight in zip(model_output, weights):
        loss += weight * Scaled_EPE_Loss(out, target_flow)
    return loss

def EPE_Loss(pred_flow, target_flow):
    b, _, h, w = target_flow.size()
    upsampled_prediction = F.interpolate(pred_flow, (h, w), mode='bilinear', align_corners=False)
    return torch.norm(target_flow - upsampled_prediction, 2, 1).mean()