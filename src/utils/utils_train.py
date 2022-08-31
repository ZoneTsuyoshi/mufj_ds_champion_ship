import json
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics as tm
import pytorch_lightning as pl
from transformers import AutoModel, AutoConfig, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from .utils_loss import get_loss_fn
from .utils_at import *


class SequenceClassification(nn.Module):
    def __init__(self, model_name:str, dropout:float=0., design_dim:int=10, hidden_dim:int=50, hidden_layers:int=2, activation_class=nn.ReLU, mlm_path:str=None):
        super().__init__()
        bert_config = AutoConfig.from_pretrained(model_name)
        if mlm_path is None:
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            self.bert = AutoModel.from_pretrained(mlm_path)
        
        self.mlp = [nn.Linear(design_dim, hidden_dim)]
        for _ in range(hidden_layers-1):
            self.mlp += [activation_class(), nn.Linear(hidden_dim, hidden_dim)]
        self.mlp = nn.Sequential(*self.mlp)
        
        self.final = nn.Linear(bert_config.hidden_size + hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, design_var, token_type_ids=None):
        bout = self.bert(input_ids, attention_mask, token_type_ids)[0][:,0]
        dout = self.mlp(design_var)
        return self.final(self.dropout(torch.cat([bout, dout], dim=-1)))[:,0]
        


class LitBertForSequenceClassification(pl.LightningModule):
    def __init__(self, model_name:str, dirpath, lr:float=0.001, design_dim:int=10, hidden_dim:int=50, hidden_layers:int=2, 
                 activation_class:str="ReLU", dropout:float=0., weight_decay:float=0.01, 
                 beta1:float=0.9, beta2:float=0.99, epsilon:float=1e-8, gradient_clipping:float=1.0,
                 loss:str="CEL", gamma:float=1, alpha:float=1, lb_smooth:float=0.1, weight:torch.tensor=None,
                 scheduler:str=None, num_warmup_steps:int=100, num_training_steps:int=1000,
                 mlm_path:str=None,
                 at:str=None, adv_lr:float=1e-4, adv_eps:float=1e-2, adv_start_epoch:int=1, adv_steps:int=1,
                 fold_id:int=0):
        super().__init__()
        self.save_hyperparameters()

        # load BERT model
        self.sc_model = SequenceClassification(model_name, dropout, design_dim, hidden_dim, hidden_layers, getattr(nn, activation_class), mlm_path)
        self.metrics = tm.MetricCollection([tm.AUROC(num_classes=1, multiclass=False), tm.F1Score(num_classes=1, multiclass=False)])
        self.loss_fn = get_loss_fn(loss, gamma, alpha, lb_smooth, weight)
        self.adversarial_training = at is not None
        if self.adversarial_training:
            self.at = eval(at.upper())(self.sc_model, self.loss_fn, adv_lr, adv_eps, adv_start_epoch, adv_steps)
            self.automatic_optimization = False
        
        
        
    def forward(self, input_ids, attention_mask, design_var, token_type_ids=None, labels=None):
        logits = self.sc_model(input_ids, attention_mask, design_var, token_type_ids)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
        return loss, logits
        

    def training_step(self, batch, batch_idx):
        if self.adversarial_training:
            loss, logits = self._adversarial_training_step(batch, batch_idx)
        else:
            loss, logits = self._shared_loss(batch, batch_idx, "train")
        self._shared_eval(batch, logits, "train")
        return loss
        

    def validation_step(self, batch, batch_idx):
        val_loss, logits = self._shared_loss(batch, batch_idx, "valid")
        self._shared_eval(batch, logits, "valid")
        
        
    def _adversarial_training_step(self, batch, batch_idx):
        opt = self.optimizers(use_pl_optimizer=True)
        loss, logits = self(**batch)
        opt.zero_grad()
        self.manual_backward(loss)

        if self.current_epoch >= self.hparams.adv_start_epoch:
            adv_loss = self.at.attack_backward(**batch, optimizer=opt, epoch=self.current_epoch)
            self.manual_backward(adv_loss)
            self.at._restore()
        
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.hparams.gradient_clipping)
        opt.step()
        self.log(f'train_loss{self.hparams.fold_id}', loss, on_step=True, on_epoch=True, logger=True)
        sch = self.lr_schedulers()
        sch.step()
        lr = float(sch.get_last_lr()[0])
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss, logits
        
        
    
    def _shared_loss(self, batch, batch_idx, prefix):
        loss, logits = self(**batch)
        self.log(f'{prefix}_loss{self.hparams.fold_id}', loss)
        return loss, logits
    
        
    def _shared_eval(self, batch, logits, prefix):
        labels = batch["labels"]
        labels_predicted = (logits >= 0.5).int()
        records = self.metrics(labels_predicted, labels)
        self.log_dict({f"{prefix}_{k}{self.hparams.fold_id}":v for k,v in records.items()}, prog_bar=False, logger=True, on_epoch=True)
        
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, logits = self(**batch)
        return F.sigmoid(logits)
    
    
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
                                     betas=(self.hparams.beta1, self.hparams.beta2), eps=self.hparams.epsilon)
        if self.hparams.scheduler is None:
            return optimizer
        elif self.hparams.scheduler in ["LSW", "LSwW", "linear", "Linear"]:
            scheduler = get_linear_schedule_with_warmup(optimizer, self.hparams.num_warmup_steps, self.hparams.num_training_steps)
        elif self.hparams.scheduler in ["CSW", "CSwW", "cosine", "Cosine"]:
            scheduler = get_cosine_schedule_with_warmup(optimizer, self.hparams.num_warmup_steps, self.hparams.num_training_steps)
        return [optimizer], [scheduler]
    
    
        