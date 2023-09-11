""" This module defines losses for hierarchical contrastive loss """
import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Dict, Optional, Union


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def common_step(self, preds, targets):
        losses = self.ce(preds, targets)
        return losses

    def forward(self, preds, targets, indices):
        losses = self.common_step(preds, targets)
        return torch.mean(losses)


class FloodCELoss(CELoss):
    def __init__(self, flood_level: float = 0.0):
        super().__init__()
        self.flood_level = flood_level

    def forward(self, preds, targets, indices):
        losses = self.common_step(preds, targets)
        if self.training:
            adjusted_loss = (torch.mean(losses) - self.flood_level).abs() + self.flood_level
        else:
            adjusted_loss = torch.mean(losses)
        return adjusted_loss


class IFloodCELoss(CELoss):
    def __init__(self, flood_level: float = 0.0):
        super().__init__()
        self.flood_level = flood_level

    def forward(self, preds, targets, indices):
        losses = self.common_step(preds, targets)
        if self.training:
            adjusted_loss = torch.mean((losses - self.flood_level).abs() + self.flood_level)
        else:
            adjusted_loss = torch.mean(losses)
        return adjusted_loss


class AdaFloodCELoss(CELoss):
    def __init__(self, gamma: float, aux_prob_dict: dict):
        super().__init__()
        self.gamma = gamma
        self.aux_prob_dict = aux_prob_dict

    def forward(self, preds, targets, indices):
        losses = self.common_step(preds, targets)
        if self.training:
            aux_probs = torch.from_numpy(np.stack([
                self.aux_prob_dict[idx.item()] for idx in indices])).to(preds.device)
            true_class_aux_probs = aux_probs[torch.arange(targets.shape[0]), targets]

            inter_aux_probs = (1-self.gamma) * true_class_aux_probs + self.gamma
            aux_losses = -torch.log(inter_aux_probs)
            #aux_losses = self.ce(inter_aux_probs, targets)

            aux_adjusted_losses = (
                losses - aux_losses).abs() + aux_losses
            adjusted_loss = torch.mean(aux_adjusted_losses)
        else:
            adjusted_loss = torch.mean(losses)
        return adjusted_loss




class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def common_step(self, preds, targets):
        losses = self.mse(preds, targets)
        return losses

    def forward(self, preds, targets, indices):
        losses = self.common_step(preds, targets)
        return torch.mean(losses)


class FloodMSELoss(MSELoss):
    def __init__(self, flood_level: float = 0.0):
        super().__init__()
        self.flood_level = flood_level

    def forward(self, preds, targets, indices):
        losses = self.common_step(preds, targets)
        if self.training:
            adjusted_loss = (torch.mean(losses) - self.flood_level).abs() + self.flood_level
        else:
            adjusted_loss = torch.mean(losses)
        return adjusted_loss


class IFloodMSELoss(MSELoss):
    def __init__(self, flood_level: float = 0.0):
        super().__init__()
        self.flood_level = flood_level

    def forward(self, preds, targets, indices):
        losses = self.common_step(preds, targets)
        if self.training:
            adjusted_loss = torch.mean((losses - self.flood_level).abs() + self.flood_level)
        else:
            adjusted_loss = torch.mean(losses)
        return adjusted_loss


class AdaFloodMSELoss(MSELoss):
    def __init__(self, gamma: float, aux_pred_dict: dict):
        super().__init__()
        self.gamma = gamma
        self.aux_pred_dict = aux_pred_dict
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, targets, indices):
        losses = self.common_step(preds, targets)
        if self.training:
            aux_preds = torch.from_numpy(np.stack([
                self.aux_pred_dict[idx.item()] for idx in indices])).to(preds.device)
            inter_aux_preds = (1-self.gamma) * aux_preds + self.gamma * targets
            aux_losses = self.mse(inter_aux_preds, targets)

            aux_adjusted_losses = (
                losses - aux_losses).abs() + aux_losses
            adjusted_loss = torch.mean(aux_adjusted_losses)
        else:
            adjusted_loss = torch.mean(losses)
        return adjusted_loss

class HingeMSELoss(MSELoss):
    def __init__(self, gamma: float, aux_pred_dict: dict):
        super().__init__()
        self.gamma = gamma
        self.aux_pred_dict = aux_pred_dict
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, preds, targets, indices):
        losses = self.common_step(preds, targets)
        if self.training:
            aux_preds = torch.from_numpy(np.stack([
                self.aux_pred_dict[idx.item()] for idx in indices])).to(preds.device)
            inter_aux_preds = (1-self.gamma) * aux_preds + self.gamma * targets
            aux_losses = self.mse(inter_aux_preds, targets)

            aux_adjusted_losses = torch.where(
                losses >= aux_losses,
                losses,
                torch.zeros_like(losses))
            adjusted_loss = torch.mean(aux_adjusted_losses)
        else:
            adjusted_loss = torch.mean(losses)
        return adjusted_loss


#class TPPLoss(nn.Module):
#    def __init__(self, num_classes: int):
#        super(TPPLoss, self).__init__()
#        self.num_classes = num_classes
#
#    def common_step(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) -> Dict[str, torch.Tensor]:
#        # compute nll
#        event_ll, surv_ll, kl, cls_ll= (
#            output_dict[constants.EVENT_LL], output_dict[constants.SURV_LL],
#            output_dict[constants.KL], output_dict[constants.CLS_LL])
#        losses = -(event_ll + surv_ll)
#
#        if cls_ll is not None:
#            losses += -cls_ll # NOTE: negative ll
#
#        if kl is not None:
#            losses += kl
#
#        return losses
#
#    def forward(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) ->  Dict[str, torch.Tensor]:
#        losses = self.common_step(output_dict, input_dict)
#        return {constants.LOSS: torch.sum(losses), constants.LOSSES: losses}
#
#
#class FloodTPPLoss(TPPLoss):
#    def __init__(self, num_classes: int, flood_level: float = 0.0):
#        super().__init__(num_classes)
#        self.flood_level = flood_level
#
#    def forward(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) -> Dict[str, torch.Tensor]:
#        losses = self.common_step(output_dict, input_dict)
#        if self.training:
#            masks = input_dict[constants.MASKS].bool()
#            flood_level = masks.sum() * self.flood_level
#            adjusted_loss = (torch.sum(losses) - flood_level).abs() + flood_level
#        else:
#            adjusted_loss = torch.sum(losses)
#        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}
#
#
#class IFloodTPPLoss(TPPLoss):
#    def __init__(self, num_classes: int, flood_level: float = 0.0):
#        super().__init__(num_classes)
#        self.flood_level = flood_level
#
#    def forward(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) -> Dict[str, torch.Tensor]:
#        losses = self.common_step(output_dict, input_dict)
#        if self.training:
#            masks = input_dict[constants.MASKS].bool()
#            flood_level = masks.sum(dim=1).view(-1) * self.flood_level
#            adjusted_loss = torch.sum((losses - flood_level).abs() + flood_level)
#        else:
#            adjusted_loss = torch.sum(losses)
#        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}
#
#
#class AdaFloodTPPLoss(TPPLoss):
#    def __init__(self, num_classes: int, alpha_init: float = 1.0,
#                 beta_init: float = 0.0, affine_train: str = None, upper_bound: float = None):
#        super().__init__(num_classes)
#
#        if affine_train is None:
#            self.alpha = torch.tensor(alpha_init)
#            self.beta = torch.tensor(beta_init)
#        elif affine_train == 'alpha':
#            self.alpha = nn.Parameter(torch.tensor(alpha_init))
#            self.beta = torch.tensor(beta_init)
#        elif affine_train == 'beta':
#            self.alpha = torch.tensor(alpha_init)
#            self.beta = nn.Parameter(torch.tensor(beta_init))
#        elif affine_train == 'both':
#            self.alpha = nn.Parameter(torch.tensor(alpha_init))
#            self.beta = nn.Parameter(torch.tensor(beta_init))
#        else:
#            raise NotImplementedError(f'affine_train: {affine_train} is not implemented')
#
#        #self.limit_const = -10000
#        #self.upper_bound = upper_bound
#        #if self.upper_bound is None:
#        #    self.upper_bound = np.inf
#
#    def aux_step(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict, aux_type: str
#    ) -> Dict[str, torch.Tensor]:
#        return output_dict[constants.AUX_LOSSES]
#
#    def forward(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) -> Dict[str, torch.Tensor]:
#
#        losses = self.common_step(output_dict, input_dict)
#        if self.training:
#            aux_losses = self.aux_step(output_dict, input_dict, 'aux')
#            trans_aux_losses = self.alpha * aux_losses + self.beta
#            #trans_aux_losses = torch.where(
#            #    trans_aux_losses > self.upper_bound,
#            #    torch.zeros_like(trans_aux_losses) * self.limit_const, trans_aux_losses)
#
#            aux_adjusted_losses = (losses - trans_aux_losses).abs() + trans_aux_losses
#            adjusted_loss = torch.sum(aux_adjusted_losses)
#        else:
#            adjusted_loss = torch.sum(losses)
#
#        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}
#
#
#    #def forward(
#    #    self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    #) -> Dict[str, torch.Tensor]:
#
#    #    losses = self.common_step(output_dict, input_dict)
#    #    if self.training:
#    #        aux1_losses = self.aux_step(output_dict, input_dict, 'aux1')
#    #        aux2_losses = self.aux_step(output_dict, input_dict, 'aux2')
#
#    #        # compute loss based on first_half bool
#    #        if len(input_dict['is_first_half']) > 0:
#    #            is_first_half = input_dict['is_first_half']
#    #            is_second_half = torch.logical_not(is_first_half)
#
#    #            trans_aux1_losses = self.alpha * aux1_losses[is_second_half] + self.beta
#    #            trans_aux2_losses = self.alpha * aux2_losses[is_first_half] + self.beta
#
#    #            aux1_adjusted_losses = (
#    #                losses[is_second_half] - trans_aux1_losses).abs() + trans_aux1_losses
#    #            aux2_adjusted_losses = (
#    #                losses[is_first_half] - trans_aux2_losses).abs() + trans_aux2_losses
#    #            adjusted_loss = torch.sum(aux1_adjusted_losses) + torch.sum(aux2_adjusted_losses)
#    #        else:
#    #            adjusted_loss = torch.sum(losses)
#    #    else:
#    #        adjusted_loss = torch.sum(losses)
#
#    #    return {constants.LOSS: adjusted_loss}
#
#
#class CLSLoss(nn.Module):
#    def __init__(self, num_classes: int):
#        super(CLSLoss, self).__init__()
#        self.num_classes = num_classes
#        self.ce = nn.CrossEntropyLoss(reduction='none')
#
#    def common_step(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) -> Dict[str, torch.Tensor]:
#        logits, labels = output_dict[constants.LOGITS], input_dict[constants.LABELS]
#        losses = self.ce(logits, labels)
#        return losses
#
#    def forward(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) ->  Dict[str, torch.Tensor]:
#        losses = self.common_step(output_dict, input_dict)
#        return {constants.LOSS: torch.mean(losses), constants.LOSSES: losses}
#
#
#class FloodCLSLoss(CLSLoss):
#    def __init__(self, num_classes: int, flood_level: float = 0.0):
#        super().__init__(num_classes)
#        self.flood_level = flood_level
#
#    def forward(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) -> Dict[str, torch.Tensor]:
#        losses = self.common_step(output_dict, input_dict)
#        if self.training:
#            adjusted_loss = (torch.mean(losses) - self.flood_level).abs() + self.flood_level
#            #import IPython; IPython.embed()
#        else:
#            adjusted_loss = torch.mean(losses)
#        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}
#
#
#class IFloodCLSLoss(CLSLoss):
#    def __init__(self, num_classes: int, flood_level: float = 0.0):
#        super().__init__(num_classes)
#        self.flood_level = flood_level
#
#    def forward(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) -> Dict[str, torch.Tensor]:
#        losses = self.common_step(output_dict, input_dict)
#        if self.training:
#            adjusted_loss = torch.mean((losses - self.flood_level).abs() + self.flood_level)
#        else:
#            adjusted_loss = torch.mean(losses)
#        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}
#
#
#class AdaFloodCLSLoss(CLSLoss):
#    def __init__(self, num_classes: int, alpha_init: float = 1.0,
#                 beta_init: float = 0.0, affine_train: str = None,
#                 gamma: float = 0.5):
#        super().__init__(num_classes)
#
#        if affine_train is None:
#            self.alpha = torch.tensor(alpha_init)
#            self.beta = torch.tensor(beta_init)
#        elif affine_train == 'alpha':
#            self.alpha = nn.Parameter(torch.tensor(alpha_init))
#            self.beta = torch.tensor(beta_init)
#        elif affine_train == 'beta':
#            self.alpha = torch.tensor(alpha_init)
#            self.beta = nn.Parameter(torch.tensor(beta_init))
#        elif affine_train == 'both':
#            self.alpha = nn.Parameter(torch.tensor(alpha_init))
#            self.beta = nn.Parameter(torch.tensor(beta_init))
#        else:
#            raise NotImplementedError(f'affine_train: {affine_train} is not implemented')
#
#        #self.flood_level = flood_level
#        #self.upper_bound = -np.log(upper_bound_prob) # 0.7 with 0.45: 74.31
#        self.gamma = gamma
#
#    def aux_step(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict, aux_type: str
#    ) -> Dict[str, torch.Tensor]:
#        losses = output_dict[constants.AUX_LOSSES]
#        logits, labels = (
#            output_dict[constants.AUX_LOGITS], input_dict[constants.LABELS])
#        #losses = self.ce(logits, labels.float())
#        #if self.training:
#        #    import IPython; IPython.embed()
#        corrects = torch.eq(torch.argmax(logits, dim=1), labels)
#        return losses, corrects
#
#    def forward(
#        self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    ) -> Dict[str, torch.Tensor]:
#
#        losses = self.common_step(output_dict, input_dict)
#        if self.training:
#            probs = torch.softmax(output_dict[constants.AUX_LOGITS], dim=1)
#            aux_losses, corrects = self.aux_step(output_dict, input_dict, 'aux')
#            trans_aux_losses = self.alpha * aux_losses + self.beta
#
#            trans_aux_losses = -torch.log(
#                (1-self.gamma) * torch.exp(-trans_aux_losses) + self.gamma)
#
#            #inter_aux_losses = -torch.log(
#            #    (1-self.gamma) * torch.exp(-trans_aux_losses) + self.gamma)
#            #trans_aux_losses = torch.where(
#            #    corrects, trans_aux_losses, inter_aux_losses)
#
#            #trans_aux_losses = torch.where(
#            #    corrects, trans_aux_losses / 100, torch.ones_like(trans_aux_losses) * 0.45) # / 10, 0.45, / 100, 0.45, / 10, / 100
#            #import IPython; IPython.embed()
#            #trans_aux_losses = torch.where(
#            #    trans_aux_losses > self.upper_bound,
#            #    torch.ones_like(trans_aux_losses) * self.flood_level, trans_aux_losses)
#
#            aux_adjusted_losses = (
#                losses - trans_aux_losses).abs() + trans_aux_losses
#            adjusted_loss = torch.mean(aux_adjusted_losses)
#        else:
#            adjusted_loss = torch.mean(losses)
#
#        return {constants.LOSS: adjusted_loss, constants.LOSSES: losses}
#
#
#    #def forward(
#    #    self, output_dict: Union[Dict, torch.Tensor], input_dict: Dict
#    #) -> Dict[str, torch.Tensor]:
#
#    #    losses = self.common_step(output_dict, input_dict)
#    #    if self.training:
#    #        aux1_losses = self.aux_step(output_dict, input_dict, 'aux1')
#    #        aux2_losses = self.aux_step(output_dict, input_dict, 'aux2')
#
#    #        # compute loss based on first_half bool
#    #        if len(input_dict['is_first_half']) > 0:
#    #            is_first_half = input_dict['is_first_half']
#    #            is_second_half = torch.logical_not(is_first_half)
#
#    #            trans_aux1_losses = self.alpha * aux1_losses[is_second_half] + self.beta
#    #            trans_aux2_losses = self.alpha * aux2_losses[is_first_half] + self.beta
#
#    #            aux1_adjusted_losses = (
#    #                losses[is_second_half] - trans_aux1_losses).abs() + trans_aux1_losses
#    #            aux2_adjusted_losses = (
#    #                losses[is_first_half] - trans_aux2_losses).abs() + trans_aux2_losses
#    #            adjusted_loss = torch.mean(aux1_adjusted_losses) + torch.mean(aux2_adjusted_losses)
#    #        else:
#    #            adjusted_loss = torch.mean(losses)
#    #    else:
#    #        adjusted_loss = torch.mean(losses)
#
#    #    return {constants.LOSS: adjusted_loss}


