import pytorch_lightning as pl
import torch
import time
from tqdm.notebook import tqdm


class DummyModule(pl.LightningModule): 
    def training_step(self, batch, batch_idx):
        #print(f'running step: epoch {self.current_epoch}, batch {batch_idx}')
        time.sleep(0.2)

    def train_dataloader(self): 
        return [torch.randn(10, 10)] * 10
    
    def configure_optimizers(self):
        return torch.optim.Adam([torch.nn.parameter.Parameter(torch.randn(10, 10))])


class SimulateInterrupt(pl.Callback): 
    def __init__(self, epoch, batch): 
        self.epoch = epoch
        self.batch = batch

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, unused: int = 0) -> None:
        if batch_idx == self.batch and pl_module.current_epoch == self.epoch: 
            raise KeyboardInterrupt()


class CheckpointAlways(pl.Callback): 
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trainer.save_checkpoint('test.ckpt')


class DummyOnlineEval(pl.Callback):
    def __init__(self) -> None:
        self.epoch = 0

    def state_dict(self):
        return {
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict) -> None:
        self.epoch = state_dict['epoch']
        print('loading')

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.current_epoch != 2: return
        print('Starting online eval')
        while self.epoch < 4: 
            trainer.save_checkpoint('test.ckpt')
            for batch in tqdm(range(4), desc=f'epoch {self.epoch}'):
                time.sleep(0.5)
                if self.epoch == 2 and batch == 2: 
                    raise KeyboardInterrupt()
            self.epoch += 1
    
        self.epoch = 0
    