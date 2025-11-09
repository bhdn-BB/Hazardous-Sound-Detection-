import time
import lightning


class LitTrainer(lightning.LightningModule):
    def __init__(
        self,
        model,
        forward,
        optimizer,
        scheduler,
        scheduler_params,
        batch_key,
        metric_input_key,
        metric_output_key,
        val_metrics,
        train_metrics,
    ):
        super().__init__()

        self.model = model
        self._forward = forward
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params
        self._batch_key = batch_key

        self._metric_input_key = metric_input_key
        self._metric_output_key = metric_output_key
        self._val_metrics = val_metrics
        self._train_metrics = train_metrics

    def _aggregate_outputs(self, losses, inputs, outputs):
        united = losses
        united.update({"input_" + k: v for k, v in inputs.items()})
        united.update({"output_" + k: v for k, v in outputs.items()})
        return united

    def training_step(self, batch):
        start_time = time.time()
        losses, inputs, outputs = self._forward(self, batch, epoch=self.current_epoch)
        model_time = time.time() - start_time

        for k, v in losses.items():
            self.log(
                "train/" + k,
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=inputs[self._batch_key].shape[0],
                sync_dist=True,
            )
            self.log(
                "train/avg_" + k,
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=inputs[self._batch_key].shape[0],
                sync_dist=True,
            )
        self.log(
            "train/model_time",
            model_time,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "train/avg_model_time",
            model_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        return self._aggregate_outputs(losses, inputs, outputs)

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        losses, inputs, outputs = self._forward(self, batch, epoch=self.current_epoch)
        model_time = time.time() - start_time

        if self._val_metrics is not None:
            self._val_metrics.update(
                outputs[self._metric_output_key],
                inputs[self._metric_input_key]
            )

        for k, v in losses.items():
            self.log(
                "valid/" + k,
                v,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                batch_size=inputs[self._batch_key].shape[0],
                sync_dist=True,
            )
            self.log(
                "valid/avg_" + k,
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=inputs[self._batch_key].shape[0],
                sync_dist=True,
            )
        self.log(
            "valid/model_time",
            model_time,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        self.log(
            "valid/avg_model_time",
            model_time,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

        return self._aggregate_outputs(losses, inputs, outputs)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        metric_values = self._val_metrics.compute()
        self.log_dict(
            {"valid/"+k:v for k,v in metric_values.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self._val_metrics.reset()

    def configure_optimizers(self):
        scheduler = {"scheduler": self._scheduler}
        scheduler.update(self._scheduler_params)
        return (
            [self._optimizer], [scheduler],
        )