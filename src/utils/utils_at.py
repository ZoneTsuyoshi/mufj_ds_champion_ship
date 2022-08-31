import torch
from torch import nn
from .utils_loss import get_loss_fn


class AT:
    def __init__(self, model, loss_fn, adv_lr:float=1.0, adv_eps:float=0.01, start_epoch:float=0, adv_step:float=1, adv_param:str="weight"):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.criterion = loss_fn


class AWP(AT):
    def __init__(self, model, loss_fn, adv_lr=1.0, adv_eps=0.01, start_epoch=0, adv_step=1, adv_param="weight"):
        super(AWP, self).__init__(model, loss_fn, adv_lr, adv_eps, start_epoch, adv_step, adv_param)
        self.backup_eps = {}
        

    def attack_backward(self, optimizer, input_ids, attention_mask, token_type_ids=None, labels=None, epoch=0):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for _ in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                logits = self.model(input_ids, attention_mask, token_type_ids)
                adv_loss = self.criterion(logits, labels)
            optimizer.zero_grad()
        return adv_loss
    

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
        
        
        
        
class FGM(AT):
    def __init__(self, model, loss_fn, adv_lr=1.0, adv_eps=0.01, start_epoch=0, adv_step=1, adv_param="word_embeddings"):
        super(FGM, self).__init__(model, loss_fn, adv_lr, adv_eps, start_epoch, adv_step, adv_param)

    
    def attack_backward(self, input_ids, attention_mask, token_type_ids=None, labels=None, optimizer=None, epoch=0):
        self._attack()
        logits = self.model(input_ids, attention_mask, token_type_ids)
        return self.criterion(logits, labels)
        
        
    def _attack(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.adv_lr * param.grad / (norm + e)
                    param.data.add_(r_at)

                    
    def _restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}
    