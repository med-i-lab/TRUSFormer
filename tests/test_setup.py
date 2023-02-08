from gc import callbacks
import pytest


def test_basic(mnist_lit_module, mnist_datamodule):

    from pytorch_lightning import Trainer

    Trainer(max_epochs=1).fit(mnist_lit_module, mnist_datamodule)


def test_interrupt_callback(
    mnist_lit_module, mnist_datamodule, interrupt_training_callback
):
    with pytest.raises(RuntimeError):

        from pytorch_lightning import Trainer

        Trainer(max_epochs=1, callbacks=[interrupt_training_callback]).fit(
            mnist_lit_module, mnist_datamodule
        )


def test_wandb_logging(mnist_lit_module, mnist_datamodule):
    from pytorch_lightning.loggers.wandb import WandbLogger

    logger = WandbLogger("test", id="test_run_0000", project="debugging")

    from pytorch_lightning import Trainer

    Trainer(max_epochs=1, logger=logger).fit(mnist_lit_module, mnist_datamodule)
