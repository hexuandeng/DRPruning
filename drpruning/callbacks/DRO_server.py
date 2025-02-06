import os
import sys
import torch
import math
import pickle
import base64
import argparse
import threading
from time import time
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)
scaling_law = None
mem = {}

class ScalingLaw(torch.nn.Module):
    def __init__(self, for_prune=False, save_folder=''):
        super(ScalingLaw, self).__init__()
        self.save_folder = save_folder
        self.device = "cpu"
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
            plt.plot(input[0].cpu().numpy(), target.cpu().numpy(), label='target')
            plt.plot(input[0].cpu().numpy(), self.forward(input).detach().cpu().numpy(), label='predict')
            plt.legend()
            plt.savefig(os.path.join(self.save_folder, f'{domain}.png'), dpi=1000)
            plt.clf()


def background_task(data, update_steps):
    start = time()
    global scaling_law, mem
    input_history_bytes = base64.b64decode(data['input_history'])
    loss_history_bytes = base64.b64decode(data['loss_history'])
    finals_bytes = base64.b64decode(data['finals'])
    set_names_bytes = base64.b64decode(data['set_names'])

    input_history = pickle.loads(input_history_bytes)
    loss_history = pickle.loads(loss_history_bytes)
    finals = pickle.loads(finals_bytes)
    set_names = pickle.loads(set_names_bytes)
    baselines = [None for _ in set_names]

    for i in range(len(set_names)):
        if loss_history[i] is not None and loss_history[i].shape[0] > 10: # and update_steps >= scaling_law.max_duration * 0.1
            variables = scaling_law.optimize(input_history[i], loss_history[i], finals[i])
            scaling_law.draw_plt(input_history[i], loss_history[i], set_names[i])
            baselines[i] = min(variables[-2], loss_history[i].min().item())

    mem[update_steps] = baselines
    print(update_steps, mem[update_steps], file=sys.stderr)
    print(time() - start, file=sys.stderr)


@app.route('/process_domain', methods=['POST'])
def process_domain():
    data = request.get_json()
    update_steps = data['update_steps']

    print(update_steps, file=sys.stderr)
    threading.Thread(target=background_task, args=(data, update_steps)).start()

    return jsonify({'message': 'Task is processing in the background.'})


@app.route('/get_mem', methods=['GET'])
def get_mem():
    update_steps = request.args.get('update_steps', type=int)
    if update_steps in mem:
        return jsonify({'baselines': mem[update_steps]})
    else:
        return jsonify({'error': 'No data found for the given update_steps.'}), 404


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask background service with command line arguments.')
    parser.add_argument('--for_prune', type=bool, default=False)
    parser.add_argument('--save_folder', type=str, default='')
    parser.add_argument('--max_duration', type=str, default=48000)
    args = parser.parse_args()
    scaling_law = ScalingLaw(args.for_prune, args.save_folder)
    scaling_law.max_duration = args.max_duration
    scaling_law.train()

    app.run(host='0.0.0.0', port=5000, debug=True)
