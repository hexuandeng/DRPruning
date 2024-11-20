from transformers import TrainerCallback
import torch
import torch.distributed as dist
from typing import List
import logging

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from time import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    logger.warning('Maximum number of iterations exceeded in bisection!')
    return 0.5 * (eta_min + eta_max)


class ScalingLaw(torch.nn.Module):
    def __init__(self, for_prune=False, save_folder=''):
        super(ScalingLaw, self).__init__()
        self.save_folder = save_folder
        self.device = torch.cuda.current_device()
        self.loss_func = torch.nn.HuberLoss(reduction='mean', delta=1e-3)
        self.ranges = [
            [x * 10 for x in range(3)],
            [x for x in range(2)],
            [x / 2 for x in range(3)],
        ]
        if for_prune:
            self.ranges.append([-x / 2 for x in range(2)])

        self.search_size = 1
        for r in self.ranges:
            self.search_size *= len(r)

        self.reinit(0)

    def reinit(self, index):
        indices = self.get_indices(index)
        self._variables = torch.nn.Parameter(
            torch.tensor([self.ranges[i][indices[i]] for i in range(len(self.ranges))], 
                         dtype=torch.float32,
                         device=self.device),
            requires_grad=True)
        self.logA = self._variables[0]
        self.logE = self._variables[1]

    def get_indices(self, index):
        indices = []
        for r in reversed(self.ranges):
            index, i = divmod(index, len(r))
            indices.append(i)
        return indices[::-1]

    def forward(self, x):
        x = x.to(self.device)
        exps = torch.sum(self._variables[2: ].unsqueeze(-1) * torch.log(x), dim=0)
        return torch.exp(self._variables[0] - exps) + torch.exp(self._variables[1])

    def compute_loss(self, inputs, targets):
        outputs = self.forward(inputs)
        return self.loss_func(outputs, targets)

    @property
    def variables(self):
        variables = self._variables.detach().clone()
        variables[0] = torch.exp(variables[0])
        variables[1] = torch.exp(variables[1])
        return variables.tolist()
    
    def optimize_from_init(self, init, input, target, final=False):
        def closure():
            self.zero_grad()
            objective = self.compute_loss(input, target)
            objective.backward()
            return objective

        self.reinit(init)
        optimizer = torch.optim.LBFGS(self.parameters(), lr=1, max_iter=20)
        step = 25 if final else 10
        for _ in range(step):
            loss = optimizer.step(closure)
        return loss, init

    def optimize(self, input, target, final):
        input = input.to(self.device)
        target = target.to(self.device)

        mem = []
        for init in range(self.search_size):
            mem.append(self.optimize_from_init(init, input, target, final=False))
        mem = sorted([(i.item(), j) for i, j in mem if not torch.isnan(i)])

        asws = []
        for _, init in mem[2::-1]:
            self.optimize_from_init(init, input, target, final=True)
            predict = self.forward(final).item()
            predict = min(predict, torch.min(target).item())
            predict = max(predict, 0)
            asw = self.variables + [predict]
            check = sum([math.isnan(i) for i in asw])
            if not check:
                asws.append(self.variables + [predict])
        
        asws = [sum([it[i] for it in asws]) / len(asws) for i in range(len(asws[0]))]
        loss = [i[0] for i in mem[2::-1]]
        return asws + [sum(loss) / len(loss)]
    
    def draw_plt(self, input, target, domain):
        with torch.no_grad():
            plt.figure()
            plt.plot(input[0].cpu().numpy(), target.cpu().numpy(), label='training loss')
            plt.plot(input[0].cpu().numpy(), self.forward(input).detach().cpu().numpy(), label='predicted loss')
            plt.legend()
            plt.savefig(os.path.join(self.save_folder, f'{domain}.png'), dpi=1000)
            plt.clf()


class DRPruningCallback(TrainerCallback):
    def __init__(self, 
                 reference_loss: List[float] = None, 
                 rho: float = 0.1, 
                 ema: float = 0.1, 
                 min_prob: float = 0.2, 
                 clamp_q_to_min: int = 1, 
                 use_eval: bool = False,
                 dynamic_update_interval: int = 400,
                 dynamic_proportion: bool = False,
                 dynamic_baseline: bool = True,
                 for_prune: bool = False,
                 save_folder: str = '') -> None:
        self.device = torch.cuda.current_device()
        self._reference_loss = reference_loss.copy()
        self.reference_loss = reference_loss.copy()
        self.use_eval = use_eval
        self.for_prune = for_prune
        self.save_folder = save_folder

        self.rho = rho
        self.EMA_alpha = ema
        self.min_prob = min_prob
        self.clamp_q_to_min = clamp_q_to_min
        self.dynamic_update_interval = dynamic_update_interval
        self.dynamic_proportion = dynamic_proportion
        self.dynamic_baseline = dynamic_baseline
        
        if self.device == 0:
            logger.info(f"Target loss: {self.reference_loss}")
            logger.info(f"Pruning {for_prune}!")

    def initialize(self):
        self.tol = 1e-4

        self.clear_history = False
        self.p_train = torch.tensor(self.proportion)
        self.sum_losses = torch.zeros(self.n_domains)  # historical loss sum over category
        self.used_domain_counts = torch.zeros(self.n_domains)
        self.domain_loss = torch.zeros(self.n_domains)
        self.domain_counts = torch.zeros(self.n_domains)

        if self.dynamic_baseline:
            self.scaling_law = ScalingLaw(self.for_prune, self.save_folder)
            self.scaling_law.train()

            self.input_history = [None] * self.n_domains
            self.loss_history = [None] * self.n_domains
            self.finals = {}

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            train_dataset = kwargs["train_dataloader"].dataset
        except:
            train_dataset = kwargs["eval_dataloader"].dataset

        self.domains = train_dataset.domains
        self.proportion = train_dataset.proportion
        self.n_domains = len(self.domains)

        self.group_losses = torch.zeros(len(train_dataset.domains), device=self.device)
        self.group_counts = torch.zeros(len(train_dataset.domains), device=self.device)
        self.eval_group_losses = torch.zeros(len(train_dataset.domains), device=self.device)
        self.eval_group_counts = torch.zeros(len(train_dataset.domains), device=self.device)

        self.initialize()

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
            for domain, loss in zip(self.domains, reduce_group_losses):
                logger.info(f"{domain} evaluation loss: {loss.item()}")

        if self.use_eval:
            self.sum_losses = reduce_group_losses.to('cpu')
            self.update_proportion(state, kwargs["train_dataloader"])

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
        self.used_domain_counts += group_counts
        self.domain_counts += group_counts
        self.domain_loss += reduce_group_losses

        self.group_losses.zero_()
        self.group_counts.zero_()
        if self.use_eval:
            return
        
        group_denom = group_counts + 1e-8
        reduce_group_losses = reduce_group_losses / group_denom

        valid_index = torch.logical_and(reduce_group_losses.ne(0), torch.logical_not(reduce_group_losses.isnan()))
        valid_losses = self.sum_losses[valid_index]
        self.sum_losses[valid_index] = valid_losses.mul(1 - self.EMA_alpha).add(reduce_group_losses[valid_index], alpha=self.EMA_alpha)
        
        if not self.use_eval and state.global_step % self.dynamic_update_interval == 0:
            self.update_proportion(state, kwargs["train_dataloader"])

    def update_proportion(self, state, train_dataloader):
        group_denom = self.domain_counts + 1e-8
        domain_loss = self.domain_loss / group_denom
        if self.device == 0:
            logger.info(f"Global Step: {state.global_step}")
            for domain, loss in zip(self.domains, domain_loss):
                logger.info(f"{domain} training loss: {loss.item()}")
        self.domain_loss.zero_()
        self.domain_counts.zero_()
        
        # Dynamic target loss
        if self.dynamic_baseline and state.global_step >= state.max_steps * 0.02:
            start_time = time()
            if self.use_eval:
                domain_loss = self.sum_losses

            # Update Loss and Corresponding Used Data for Each Domain
            max_count = 0
            for i in range(self.n_domains):
                if domain_loss[i].ne(0):
                    if self.for_prune:
                        input_history = torch.tensor([self.used_domain_counts[i].item(), state.outputs['l0_output'][1]['expected_sparsity']]).unsqueeze(-1)
                    else:
                        input_history = torch.tensor([self.used_domain_counts[i].item(), ]).unsqueeze(-1)
                    self.finals[i] = input_history.detach().clone()
                    self.finals[i][0, 0] *= (state.max_steps / state.global_step)
                    if self.for_prune:
                        self.finals[i][-1, 0] = state.outputs['l0_output'][1]['target_sparsity']
                    max_count = max(max_count, self.finals[i][0, 0])

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
                            
            # Update Baseline Loss
            for i in range(self.n_domains):
                if self.loss_history[i] is not None and self.loss_history[i].shape[0] >= 10 and state.global_step >= state.max_steps * 0.1:
                    variables = self.scaling_law.optimize(self.input_history[i], self.loss_history[i], self.finals[i])
                    self.scaling_law.draw_plt(self.input_history[i], self.loss_history[i], self.domains[i])
                    baseline = min(variables[-2], self.loss_history[i].min().item())
                    if self.reference_loss[i] == self._reference_loss[i]:
                        self.reference_loss[i] = baseline
                    else:
                        self.reference_loss[i] = min(self.reference_loss[i], baseline)
                    if self.device == 0:
                        logger.info(f"{self.domains[i]}: {variables}")
            if self.device == 0:
                logger.info(f"Target loss: {self.reference_loss}")
                logger.info(f"Time: {time() - start_time}")

        # Standard DRO update proportion
        new_proportion = self.update_mw()
        train_dataloader.dataset.proportion = new_proportion

        # Dynamic proportion
        if self.dynamic_proportion and state.global_step >= state.max_steps * 0.2:
            times = 2
            proportion = torch.tensor(self.proportion)
            p_train = self.p_train.mul(1 - self.EMA_alpha).add(torch.tensor(new_proportion), alpha=self.EMA_alpha)
            p_train = torch.clamp(p_train, min=proportion / times, max=proportion * times)
            p_train = p_train / p_train.sum()
            p_train = torch.clamp(p_train, min=torch.min(proportion).item(), max=torch.max(proportion).item())
            self.p_train = p_train / p_train.sum()

    def update_mw(self):
        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        past_losses = self.sum_losses - torch.tensor(self.reference_loss)
        times = 10
        while past_losses.max() < 0:
            times -= 1
            past_losses = self.sum_losses - torch.tensor(self.reference_loss) * times * 0.1
        if times < 10:
            logger.warning(f"Target Loss too high! Times {times * 0.1}!")

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
            logger.warning("bisection failed! Ignore updating")
            q = p_train
        else:
            q = p(eta_star)
        if hasattr(self, 'clamp_q_to_min') and self.clamp_q_to_min:
            q = torch.clamp(q, min=torch.min(self.p_train).item())        
        q = q.tolist()
        q = [i / sum(q) for i in q]

        if self.device == 0:
            logger.info("EMA before-baseline losses: {}".format(
                " ".join(["{:.6f}".format(xx.item()) for xx in self.sum_losses[0:self.n_domains]])))
            logger.info("EMA after-baseline losses:  {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in past_losses[0:self.n_domains]])))
            logger.info("EMA group fractions: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in self.p_train[0:self.n_domains]])))
            logger.info("Group loss weights:  {}".format(" ".join(["{:.6f}".format(xx) for xx in q[0:self.n_domains]])))

        if self.clear_history:
            self.sum_losses.zero_()

        return q
