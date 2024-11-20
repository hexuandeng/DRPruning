from transformers import TrainerCallback
import torch
import logging
import torch.distributed as dist
from typing import Any, Dict, List
from transformers import TrainerCallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ShearedCallback(TrainerCallback):
    def __init__(self, 
                 reference_loss: List[float] = None, 
                 update_type: str ="sheared"
                 ) -> None:
        self.device = torch.cuda.current_device()
        self.reference_loss = reference_loss
        self.update_type = update_type
        
        if self.device == 0:
            logger.info(f"Target loss: {self.reference_loss}")
            logger.info(f"Update type {update_type}!")

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            train_dataset = kwargs["train_dataloader"].dataset
        except:
            train_dataset = kwargs["eval_dataloader"].dataset

        self.set_names = train_dataset.domains
        self.proportion = train_dataset.proportion
        self.n_domains = len(self.set_names)
        self.group_losses = torch.zeros(len(train_dataset.domains), device=self.device)
        self.group_counts = torch.zeros(len(train_dataset.domains), device=self.device)
        self.eval_group_losses = torch.zeros(len(train_dataset.domains), device=self.device)
        self.eval_group_counts = torch.zeros(len(train_dataset.domains), device=self.device)
        self.domain_loss = torch.zeros(self.n_domains)
        self.domain_counts = torch.zeros(self.n_domains)
        
        if self.device == 0:
            logger.info("Group loss weights:  {}".format(" ".join(["{:.6f}".format(xx) for xx in train_dataset.proportion[0:self.n_domains]])))

    def on_evaluate(self, args, state, control, **kwargs):
        group_losses = self.eval_group_losses
        group_counts = self.eval_group_counts
        reduce_group_losses = group_losses.detach().clone()
        if torch.cuda.device_count() > 1:
            dist.all_reduce(group_counts)
            dist.all_reduce(reduce_group_losses)
        
        reduce_group_losses = reduce_group_losses / group_counts
        if self.device == 0:
            for domain, loss in zip(self.set_names, reduce_group_losses):
                logger.info(f"{domain} evaluation loss: {loss.item()}")

        try:
            self.update_proportion(state, kwargs["train_dataloader"], reduce_group_losses.to('cpu'))
        except:
            self.update_proportion(state, kwargs["eval_dataloader"], reduce_group_losses.to('cpu'))
        self.eval_group_losses.zero_()
        self.eval_group_counts.zero_()

    def on_step_end(self, args, state, control, **kwargs):
        group_losses = self.group_losses
        group_counts = self.group_counts
        reduce_group_losses = group_losses.detach().clone()
        if torch.cuda.device_count() > 1:
            dist.all_reduce(group_counts)
            dist.all_reduce(reduce_group_losses)

        group_counts = group_counts.cpu()
        reduce_group_losses = reduce_group_losses.cpu()
        self.domain_counts += group_counts
        self.domain_loss += reduce_group_losses

        self.group_losses.zero_()
        self.group_counts.zero_()

    def update_proportion(self, state, train_dataloader, losses):
        group_denom = self.domain_counts + 1e-8
        domain_loss = self.domain_loss / group_denom
        if self.device == 0:
            logger.info(f"Global Step: {state.global_step}")
            print(getattr(self, "l0_output", None))
            for domain, loss in zip(self.set_names, domain_loss):
                logger.info(f"{domain} training loss: {loss.item()}")
        self.domain_loss.zero_()
        self.domain_counts.zero_()
        
        current_prop = train_dataloader.dataset.proportion
        """ Update the proportion of each domain """
        diff = losses - torch.tensor(self.reference_loss)
        eta = 1.
        c = 1e-4 # following Doremi (Xie et al., 2023)
        
        if self.update_type == "sheared": # update with exponential descent
            updated_alpha = torch.log(torch.tensor(current_prop)) + eta * diff 
            updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
            updated_domain_weights = (1-c) * updated_alpha + c / self.n_domains
        elif self.update_type == "bandit": 
            updated_alpha = torch.tensor(current_prop) + eta * diff 
            updated_alpha = torch.nn.functional.softmax(updated_alpha, dim=0)
            updated_domain_weights = (1-c) * updated_alpha + c / self.n_domains
        elif self.update_type == "constant": # constant proportion
            updated_domain_weights = torch.tensor(current_prop)
            
        updated_domain_weights = updated_domain_weights.numpy().astype('float64')
        updated_domain_weights = updated_domain_weights / updated_domain_weights.sum()
        updated_domain_weights = updated_domain_weights.tolist()
        train_dataloader.dataset.proportion = updated_domain_weights
        if self.device == 0:
            logger.info("EMA before-baseline losses: {}".format(
                " ".join(["{:.6f}".format(xx.item()) for xx in losses[0:self.n_domains]])))
            logger.info("EMA after-baseline losses:  {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in diff[0:self.n_domains]])))
            logger.info("EMA group fractions: {}".format(" ".join(["{:.6f}".format(xx) for xx in self.proportion[0:self.n_domains]])))
            logger.info("Group loss weights:  {}".format(" ".join(["{:.6f}".format(xx) for xx in updated_domain_weights[0:self.n_domains]])))
