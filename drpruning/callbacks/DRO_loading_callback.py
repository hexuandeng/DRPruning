import os
import torch
import numpy as np
from composer import Callback, Logger, State
from composer.loggers import Logger
from composer.utils import dist
import matplotlib.pyplot as plt
import math
from time import time
from typing import Any, Dict, List
import pickle
import base64
import requests

def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=1000):
    """Expects f an increasing function and return eta in [eta_min, eta_max]
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta

    # if the minimum is not reached in max_iter, returns the current value
    print('Maximum number of iterations exceeded in bisection')
    return 0.5 * (eta_min + eta_max)


class DRPruningCallback(Callback):
    "Callback for DRPrunin of data from different domains."
    def __init__(self, 
                 reference_loss: List[float], 
                 proportion: List[float],
                 set_names: List[str],
                 rho: float = 0.05, 
                 ema: float = 0.1, 
                 min_prob: float = 0.2, 
                 clamp_q_to_min: int = 1, 
                 use_eval: bool = True,
                 dynamic_proportion: bool = False,
                 dynamic_baseline: bool = False,
                 for_prune: bool = False,
                 save_folder: str = '') -> None:
        self.set_names = set_names
        self.n_domains = len(set_names)
        self.reference_loss = reference_loss
        self._reference_loss = reference_loss.copy()
        self.proportion = proportion
        self.use_eval = use_eval
        self.for_prune = for_prune
        self.save_folder = save_folder
        self.used_domain_ids = [[] for _ in range(self.n_domains)]
        print("Target loss:", self.reference_loss)
        print(f"Pruning {for_prune}!")

        self.rho = rho
        self.EMA_alpha = ema
        self.min_prob = min_prob
        self.clamp_q_to_min = clamp_q_to_min
        self.dynamic_proportion = dynamic_proportion
        self.dynamic_baseline = dynamic_baseline

        self.initialize()

    def initialize(self):
        self.update_steps = 0
        self.tol = 1e-4

        self.clear_history = False
        self.device = torch.cuda.current_device()
        self.p_train = torch.tensor(self.proportion)
        self.valid_losses = torch.zeros(self.n_domains)
        self.sum_losses = torch.zeros(self.n_domains)  # historical loss sum over category
        self.count_cat = torch.ones(self.n_domains)

        if self.dynamic_baseline:
            self.used_domain_counts = torch.zeros(self.n_domains)
            self.domain_loss = torch.zeros(self.n_domains)
            self.domain_counts = torch.zeros(self.n_domains)
            self.input_history = [None] * self.n_domains
            self.loss_history = [None] * self.n_domains
            self.finals = {}
            self.to_get = None

    def after_train_batch(self, state: State, logger: Logger) -> None:
        """ Print out the number of used samples in each domain after each training batch, and log the updated proportion of each domain """
        self.update_steps += 1
        idx = state.batch["idx"]
        sets = state.batch["set"]
        all_idx = torch.cat(dist.all_gather(idx))
        all_sets = torch.cat(dist.all_gather(sets))
        dist.barrier() 
        
        for i in range(self.n_domains):
            mask = all_sets == i
            domain_idx = all_idx[mask]
            self.used_domain_ids[i].extend(domain_idx.cpu().tolist())
            # for debugging
            # print(f"domain {i} used {mask.sum().item()} new samples")

        prop = state.train_dataloader.dataset.proportion
        for domain in self.set_names:
            logger.log_metrics({f'metrics/train/{domain}_weight': round(prop[self.set_names.index(domain)], 4)})

    def before_train_batch(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`.Event.BEFORE_TRAIN_BATCH` event.

        Args:
            state (State): The training state.
            logger (Logger): The logger.
        """
        if self.to_get is not None:
            start = time()
            print(self.to_get)
            response = requests.get(f"http://localhost:5000/get_mem?update_steps={self.to_get}")
            if response.status_code == 200:
                result = response.json()
                baselines = result['baselines']
                for i in range(self.n_domains):
                    if baselines[i] is None:
                        continue
                    if self.target_loss[i] == self._target_loss[i]:
                        self.target_loss[i] = baselines[i]
                    else:
                        self.target_loss[i] = min(self.target_loss[i], baselines[i])
                
                print("Target loss:", self.target_loss)
                if sum(self._target_loss) > 0 or self.update_steps >= state.max_duration.value * 0.1:
                    self.update_proportion(state)
                self.to_get = None
            else:
                print(f"Step {self.to_get} not ready yet.")
            print("Time:", time() - start)

    def eval_end(self, state: State, logger: Logger) -> None:
        """ Update the proportion of each domain after each evaluation and update the dataset """
        if self.use_eval:
            losses = []
            for domain in self.set_names:
                losses.append(state.eval_metrics["eval"][f"{domain}_LanguageCrossEntropy"].compute().item())
            self.sum_losses = torch.tensor(losses)
        
        if self.dynamic_baseline and self.update_steps >= state.max_duration.value * 0.02:
            if self.use_eval:
                domain_loss = torch.tensor(losses)
            else:
                group_denom = self.domain_counts + 1e-8
                domain_loss = self.domain_loss / group_denom
                self.domain_loss.zero_()
                self.domain_counts.zero_()

            # Update Loss and Corresponding Used Data for Each Domain
            for i in range(self.n_domains):
                if domain_loss[i].ne(0):
                    if self.for_prune:
                        input_history = torch.tensor([self.used_domain_counts[i].item(), state.outputs['l0_output'][1]['expected_sparsity']]).unsqueeze(-1)
                    else:
                        input_history = torch.tensor([self.used_domain_counts[i].item(), ]).unsqueeze(-1)
                    self.finals[i] = input_history.detach().clone()
                    self.finals[i][0, 0] *= (state.max_duration.value / self.update_steps)
                    if self.for_prune:
                        self.finals[i][-1, 0] = state.outputs['l0_output'][1]['target_sparsity']

                    loss_history = domain_loss[i].unsqueeze(0)
                    if self.input_history[i] is None:
                        self.input_history[i] = input_history
                        self.loss_history[i] = loss_history
                    else:
                        self.input_history[i] = torch.cat((self.input_history[i], input_history), dim=-1)
                        self.loss_history[i] = torch.cat((self.loss_history[i], loss_history), dim=0)
                        if self.loss_history[i].shape[0] > 64:
                            random_index = torch.randint(0, 32, (1, )).item()
                            self.input_history[i] = torch.cat(
                                (self.input_history[i][:, :random_index],
                                 self.input_history[i][:, random_index+1:]),
                                dim=-1)
                            self.loss_history[i] = torch.cat(
                                (self.loss_history[i][:random_index],
                                 self.loss_history[i][random_index+1:]),
                                dim=0)

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if self.update_steps >= state.max_duration.value * 0.1 and rank == 0:
                data = {
                    'input_history': base64.b64encode(pickle.dumps(self.input_history)).decode('utf-8'),
                    'loss_history': base64.b64encode(pickle.dumps(self.loss_history)).decode('utf-8'),
                    'finals': base64.b64encode(pickle.dumps(self.finals)).decode('utf-8'),
                    'set_names': base64.b64encode(pickle.dumps(self.set_names)).decode('utf-8'),
                    'update_steps': self.update_steps,
                }
                requests.post('http://localhost:5000/process_domain', json=data)
                self.to_get = self.update_steps

    def update_proportion(self, state: State):
        new_proportion = self.update_mw()
        state.train_dataloader.dataset.update_proportion(new_proportion)
        if self.dynamic_proportion and self.update_steps >= state.max_duration.value * 0.2:
            EMA_alpha = 0.1
            times = 2
            proportion = torch.tensor(self.proportion)
            p_train = self.p_train.mul(1 - EMA_alpha).add(torch.tensor(new_proportion), alpha=EMA_alpha)
            p_train = p_train / p_train.sum()
            p_train = torch.clamp(p_train, min=proportion / times, max=proportion * times)
            p_train = p_train / p_train.sum()
            p_train = torch.clamp(p_train, min=torch.min(proportion).item(), max=torch.max(proportion).item())
            self.p_train = p_train / p_train.sum()
            
    def after_loss(self, state: State, logger: Logger) -> None:
        """Called on the :attr:`.Event.AFTER_LOSS` event.

        Args:
            state (State): The training state.
            logger (Logger): The logger.
        """
        group_losses = state.loss["group_losses"]
        group_counts = state.loss["group_counts"]
        reduce_group_losses = group_losses.detach().clone()
        if torch.cuda.device_count() > 1:
            dist.all_reduce(group_counts)
            dist.all_reduce(reduce_group_losses)

        group_counts = group_counts.cpu()
        reduce_group_losses = reduce_group_losses.cpu()
        if self.dynamic_baseline:
            self.used_domain_counts += group_counts
            self.domain_counts += group_counts
            self.domain_loss += reduce_group_losses
        if self.use_eval:
            return
        
        group_denom = group_counts + 1e-8
        reduce_group_losses = reduce_group_losses / group_denom

        valid_index = torch.logical_and(reduce_group_losses.ne(0), torch.logical_not(reduce_group_losses.isnan()))
        valid_losses = self.sum_losses[valid_index]
        valid_counts = self.count_cat[valid_index]
        self.sum_losses[valid_index] = valid_losses.mul(1 - self.EMA_alpha).add(reduce_group_losses[valid_index], alpha=self.EMA_alpha)
        self.count_cat[valid_index] = valid_counts.add(group_counts[valid_index])

    def update_mw(self):
        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        past_losses = self.sum_losses - torch.tensor(self.reference_loss)
        times = 10
        while past_losses.max() < 0:
            times -= 1
            past_losses = self.sum_losses - torch.tensor(self.reference_loss) * times * 0.1
        if times < 10:
            print(f"Target Loss too high! Times {times * 0.1}!")

        rho = self.rho
        p_train = self.p_train

        if hasattr(self, 'min_prob'):
            min_prob = self.min_prob
        else:
            min_prob = 0.2

        def p(eta):
            pp = p_train * torch.relu(past_losses - eta)
            q = pp / pp.sum()
            cq = torch.clamp(q / p_train, min=min_prob)
            return cq * p_train / (cq * p_train).sum()

        def bisection_target(eta):
            pp = p(eta)
            return 0.5 * ((pp / p_train - 1) ** 2 * p_train).sum() - rho

        eta_min = -(1.0 / (np.sqrt(2 * rho + 1) - 1)) * past_losses.max()
        eta_max = past_losses.max()
        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=1000)

        if eta_star == eta_max:
            print("bisection failed! Ignore updating")
            q = p_train
        else:
            q = p(eta_star)
        if hasattr(self, 'clamp_q_to_min') and self.clamp_q_to_min:
            q = torch.clamp(q, min=torch.min(self.p_train).item())        
        q = q.tolist()
        q = [i / sum(q) for i in q]

        print("EMA before-baseline losses: {}".format(
            " ".join(["{:.6f}".format(xx.item()) for xx in self.sum_losses[0:self.n_domains]])))
        print("EMA after-baseline losses:  {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in past_losses[0:self.n_domains]])))
        print("EMA group fractions: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in self.p_train[0:self.n_domains]])))
        print("Group loss weights:  {}".format(" ".join(["{:.6f}".format(xx) for xx in q[0:self.n_domains]])))

        if self.clear_history:
            self.sum_losses.zero_()
        # self.count_cat.fill_(1.)

        return q

    def state_dict(self) -> Dict[str, Any]:
        """ Save the used domain ids after each epoch, for resuming training from a previous checkpoint to make sure that used samples are not used again """
        state = {
            "update_steps": self.update_steps,
            "used_domain_ids": self.used_domain_ids,
            "valid_losses": self.valid_losses.tolist(),
            "sum_losses": self.sum_losses.tolist(),
            "count_cat": self.count_cat.tolist(),
        }
        if self.dynamic_proportion:
            state["p_train"] = self.p_train.tolist()
        if self.dynamic_baseline:
            state["reference_loss"] = self.reference_loss
            state["used_domain_counts"] = self.used_domain_counts.tolist()
            state["input_history"] = self.input_history
            state["loss_history"] = self.loss_history
            
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Load the used domain ids """
        self.update_steps = state_dict["update_steps"]
        self.used_domain_ids = state_dict["used_domain_ids"]
        self.valid_losses = torch.tensor(state_dict["valid_losses"])
        self.sum_losses = torch.tensor(state_dict["sum_losses"])
        self.count_cat = torch.tensor(state_dict["count_cat"])
        if self.dynamic_proportion:
            self.p_train = torch.tensor(state_dict["p_train"])
        if self.dynamic_baseline:
            self.reference_loss = state_dict["reference_loss"]
            self.used_domain_counts = torch.tensor(state_dict["used_domain_counts"])
            self.input_history = state_dict["input_history"] # AttributeError: 'NoneType' object has no attribute 'to'
            self.loss_history = state_dict["loss_history"]
