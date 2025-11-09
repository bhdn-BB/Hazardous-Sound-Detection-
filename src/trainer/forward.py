import torch


class AudioForward(torch.nn.Module):
    def __init__(
        self,
        loss_function,
        output_key,
        input_key,
    ):
        super().__init__()
        self.loss_function = loss_function
        self.output_key = output_key
        self.input_key = input_key

    def forward(self, runner, batch, epoch=None):
        specs, targets = batch
        output = runner.model(specs)
        output["sigmoid_predictions"] = torch.sigmoid(output["logits"])
        output["softmax_predictions"] = torch.softmax(output["logits"], dim=-1)
        inputs = {
            "specs": specs,
            "targets": targets,
            "targets_1d": targets.argmax(dim=-1),
        }
        losses = {
            "loss": self.loss_function(
                output[self.output_key],
                inputs[self.input_key],
            )
        }
        return losses, inputs, output