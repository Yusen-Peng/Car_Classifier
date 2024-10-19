import lightning as L

import torch

from torch import optim,nn,utils
import torch.nn.functional as F

class TimeSeriesClassificationModule(L.LightningModule):
    def __init__(self,classifier,loss_fn,lr=1e-3):
        super(TimeSeriesClassificationModule,self).__init__()
        self.model = classifier
        self.loss_fn = F.cross_entropy
        self.lr = lr
    def forward(self,inputs):
        return self.model(inputs)
    def training_step(self,batch,batch_idx):
        inputs,target = batch
        pred = self.model(inputs)

        #pred = F.softmax(output)
        # _ , pred = torch.topk(output,k=1,dim=1)
        #target = torch.unsqueeze(target,dim=1)

        loss = self.loss_fn(pred,target.long())

        self.log("train_loss",loss,prog_bar=True)
        #loss = F.cross_entropy(pred,output)
        return loss
    
    def predict_step(self, batch,batch_idx):
        inputs,target = batch
        output = self.model(inputs)
        # _ , pred = torch.topk(output,k=1,dim=1)
        pred = output

        return pred
    
    def test_step(self, batch,batch_idx):
        inputs,target = batch

        output = self(inputs)
        pred = output
        # _ , pred = torch.topk(output,k=1,dim=1)

        return pred

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=self.lr)
        return optimizer
