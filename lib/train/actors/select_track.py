from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_condv2, adjust_keep_rate


class SelectTrackActor(BaseActor):
    """ Actor for training TBSI_Track models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data['visible'])

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['visible']['template_images']) == 1
        assert len(data['visible']['search_images']) == 1

        template_img_v = data['visible']['template_images'][0].view(-1, *data['visible']['template_images'].shape[2:])  # (batch, 3, 128, 128)
        template_img_i = data['infrared']['template_images'][0].view(-1, *data['infrared']['template_images'].shape[2:])  # (batch, 3, 128, 128)        
        
        search_img_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_zv = None
        box_mask_zi = None
        ce_keep_rate = None
        box_mask_zv = generate_mask_condv2(self.cfg, template_img_v.shape[0], template_img_v.device,
                                            data['visible']['template_anno'][0])
        box_mask_zi = generate_mask_condv2(self.cfg, template_img_v.shape[0], template_img_v.device,
                                            data['infrared']['template_anno'][0])
        out_dict = self.net(template=[template_img_v, template_img_i],
                            search=[search_img_v, search_img_i],
                            template_mask_zv=box_mask_zv,
                            template_mask_zi=box_mask_zi)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True, entropy=False):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)        

        import torch.nn as nn
        sel_loss = pred_dict['loss_sel']
    
        
        sel_loss0 = sel_loss[0] if len(sel_loss) > 0 else 0
        sel_loss4 = sel_loss[1] if len(sel_loss) > 1 else 0
        sel_loss8 = sel_loss[2] if len(sel_loss) > 2 else 0 
               
        w1 = nn.Parameter(torch.tensor(1.0/64, requires_grad=True))
        w2 = nn.Parameter(torch.tensor(1.0/64, requires_grad=True))
        w3 = nn.Parameter(torch.tensor(1.0/64, requires_grad=True))   
        
        sel_loss = w1 * sel_loss0 + w2 * sel_loss4 + w3 * sel_loss8
               
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + sel_loss
        if entropy and pred_dict['decisions'] != []:
            epsilon = 1e-5
            prob1 = pred_dict['decisions']
            prob2 = 1 - pred_dict['decisions']
            entropy_loss = (1 + prob1 * torch.log2(prob1 + epsilon) + prob2 * torch.log2(prob2 + epsilon)).mean()
            loss += entropy_loss

        if return_status:
            # Status for log
            mean_iou = iou.detach().mean()
            if entropy and pred_dict['decisions'] != []:
                status = {'Ls/total': loss.item(),
                          'Ls/giou': giou_loss.item(),
                          'Ls/l1': l1_loss.item(),
                          'Ls/loc': location_loss.item(),
                          'Ls/entropy': entropy_loss.item(),
                          'IoU': mean_iou.item()}
            else:
                status = {'Ls/total': loss.item(),
                          'Ls/giou': giou_loss.item(),
                          'Ls/l1': l1_loss.item(),
                          'Ls/loc': location_loss.item(),
                          'IoU': mean_iou.item()}
            return loss, status
        else:
            return loss
