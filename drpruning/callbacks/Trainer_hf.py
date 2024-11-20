import logging
from transformers import Trainer

logger = logging.getLogger(__name__)


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        callback = self.callback_handler.callbacks[1]

        self.model.model.pruned_steps = self.state.global_step
        inputs["return_dict"] = True
        domain = inputs.pop("domain").detach().clone()
        return_loss, outputs = super().compute_loss(model, inputs, True)
        if 'l0_output' in outputs:
            return_loss += outputs.l0_output[0]
            callback.l0_output = outputs.l0_output[1]

        if not hasattr(callback, 'eval_group_losses'):
            self.callback_handler.on_train_begin(self.args, self.state, self.control)

        if self.model.training:
            callback.group_losses[int(domain.item())] += return_loss.item()
            callback.group_counts[int(domain.item())] += 1
        else:
            callback.eval_group_losses[int(domain.item())] += return_loss.item()
            callback.eval_group_counts[int(domain.item())] += 1

        return (return_loss, outputs) if return_outputs else return_loss
