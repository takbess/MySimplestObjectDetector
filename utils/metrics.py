import torch

def IoU(bboxes1, bboxes2):
    # bboxes1, bboxes2 is (N, 4)
    # x, y, w, h order
    # xmin, ymin, xmax, ymax transform
    bboxes1 = torch.stack([bboxes1[:,0] - bboxes1[:,2] / 2,
                           bboxes1[:,1] - bboxes1[:,3] / 2,
                           bboxes1[:,0] + bboxes1[:,2] / 2,
                           bboxes1[:,1] + bboxes1[:,3] / 2], dim=1)
    bboxes2 = torch.stack([bboxes2[:,0] - bboxes2[:,2] / 2,
                           bboxes2[:,1] - bboxes2[:,3] / 2,
                           bboxes2[:,0] + bboxes2[:,2] / 2,
                           bboxes2[:,1] + bboxes2[:,3] / 2], dim=1)
    
    # Intersection
    inter_xmin = torch.max(bboxes1[:,0], bboxes2[:,0])
    inter_ymin = torch.max(bboxes1[:,1], bboxes2[:,1])
    inter_xmax = torch.min(bboxes1[:,2], bboxes2[:,2])
    inter_ymax = torch.min(bboxes1[:,3], bboxes2[:,3])
    
    inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
    inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
    
    inter_area = inter_w * inter_h
    
    # Union
    area1 = (bboxes1[:,2] - bboxes1[:,0]) * (bboxes1[:,3] - bboxes1[:,1])
    area2 = (bboxes2[:,2] - bboxes2[:,0]) * (bboxes2[:,3] - bboxes2[:,1])
    union_area = area1 + area2 - inter_area
    
    # IoU
    iou = inter_area / union_area
    
    return iou