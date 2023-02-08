import torch
from src.modeling.vicreg import VICReg
from src.modeling.finetuning import Finetuner
from tqdm import tqdm
import wandb
import os
import logging
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

LOG_LOSS_EVERY_N_STEPS = 100


class VICRegExperiment:
    def __init__(
        self,
        dataset_name,
        backbone_name: str,
        run_id=None,
        batch_size=64,
        checkpoint_every_n_epochs: int = 3,
        checkpoint_dir: str = "checkpoints",
        num_epochs: int = 100,
        lr: float = 1e-4,
        device: str = "cuda:0",
    ):
        from src.data.registry import create_dataset

        dataset = create_dataset(dataset_name, split="train")

        self.train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        from src.modeling.registry import create_model

        self.backbone = create_model(backbone_name)

        # vicreg model to get loss
        self.vicreg = VICReg(
            self.backbone,
            proj_dims=[512, 1024],
            features_dim=self.backbone.features_dim,
        ).to(device)

        self.vicreg_optimizer = torch.optim.Adam(self.vicreg.parameters(), lr=lr)
        # perform updates every step, so num_epochs must be converted to num_steps
        # for the scheduler
        num_steps = len(self.train_loader) * num_epochs
        num_warmup_steps = int(0.1 * num_steps)
        self.vicreg_scheduler = LinearWarmupCosineAnnealingLR(
            self.vicreg_optimizer, warmup_epochs=num_warmup_steps, max_epochs=num_steps
        )

        self.epoch = 0
        self.num_epochs = num_epochs
        self.device = device

        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        wandb.init(project="vicreg", name=None, id=run_id, config=locals())
        wandb.watch(self.vicreg)

    def save_checkpoint(self):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.vicreg.state_dict(),
                "optimizer_state_dict": self.vicreg_optimizer.state_dict(),
                "scheduler_state_dict": self.vicreg_scheduler.state_dict(),
            },
            os.path.join(self.checkpoint_dir, f"checkpoint-last.pt"),
        )
        logging.info(f"Saved experiment checkpoint to {self.checkpoint_dir}")

    def save_model(self, epoch):
        torch.save(self.vicreg, os.path.join(self.checkpoint_dir, f"vicreg-{epoch}.pt"))

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            logging.info(f"Checkpoint {checkpoint_path} does not exist, skipping.")
            return
        checkpoint = torch.load(checkpoint_path)
        self.vicreg.load_state_dict(checkpoint["model_state_dict"])
        self.vicreg_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.vicreg_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        logging.info(f"Loaded checkpoint from {checkpoint_path}")

    def train_vicreg(self, epoch):
        self.vicreg.train()

        losses = {
            "loss": 0,
            "sim_loss": 0,
            "var_loss": 0,
            "cov_loss": 0,
        }
        batches_accumulated_for_loss_logging = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
            X1, X2 = batch[0]
            X1 = X1.to(self.device)
            X2 = X2.to(self.device)
            loss = self.vicreg(X1, X2)

            for k, v in loss.items():
                losses[k] += v.item()

            loss["loss"].backward()
            self.vicreg_optimizer.step()
            self.vicreg_optimizer.zero_grad()
            batches_accumulated_for_loss_logging += 1

            if batches_accumulated_for_loss_logging == LOG_LOSS_EVERY_N_STEPS:
                wandb.log(
                    {
                        f"train/{k}": v / LOG_LOSS_EVERY_N_STEPS
                        for k, v in losses.items()
                    }
                )
                losses = {k: 0 for k in losses.keys()}
                batches_accumulated_for_loss_logging = 0
                wandb.log({"train/lr": self.vicreg_optimizer.param_groups[0]["lr"]})

            self.vicreg_scheduler.step()

    def run(self):
        self.load_checkpoint(os.path.join(self.checkpoint_dir, f"checkpoint-last.pt"))
        for epoch in range(self.epoch, self.num_epochs):
            self.train_vicreg(epoch)
            self.save_checkpoint()
            if epoch % self.checkpoint_every_n_epochs == 0:
                self.save_model(epoch)


class FineTune:
    def __init__(self, lr, epochs, checkpoint_path, device):
        self.lr = lr
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.device = device

    def run(self):
        ...
