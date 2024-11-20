import torch
from torch import nn
from argparse import Namespace as NS

from composer.core.time import Time
from drpruning.models.l0_module import Mask

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6

class L0Module(nn.Module):
    def __init__(self, cfg, device):
        super(L0Module, self).__init__()

        # base and target model info
        n_matrix_mlp = 2 if "pythia" in cfg.name else 3
        self.base_model_info = self.set_model_info(cfg, n_matrix_mlp=n_matrix_mlp) 
        l0_module_cfg = cfg.l0_module
        self.target_model_info = None
        target_model_cfg = getattr(l0_module_cfg, "target_model", None)
        if target_model_cfg is not None:
            self.target_model_info = self.set_model_info(target_model_cfg, n_matrix_mlp=n_matrix_mlp)
        
        # l0 config
        self.pruning_modules = l0_module_cfg.pruning_modules        
        self.start_sparsity = l0_module_cfg.start_sparsity 
        self.lagrangian_warmup_steps = Time.from_timestring(l0_module_cfg.lagrangian_warmup_steps).value
        self.device = device
        self.eval_target_model = l0_module_cfg.get("eval_target_model", True)
        
        # l0 params
        self.lambdas = {}
        self.lambdas["lambda_1"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.lambdas["lambda_2"] = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.masks = {}
        for pruning_module in self.pruning_modules:
            self.initialize_one_module(pruning_module)
        self.masks = torch.nn.ModuleDict(self.masks)
        self.lambdas = torch.nn.ParameterDict(self.lambdas)
        
        # config after initialization
        self.prunable_model_size = self.calculate_prunable_model_size(self.base_model_info)
        self.prunable_target_model_size = self.calculate_prunable_model_size(self.target_model_info)
        self.target_sparsity = 1 - self.prunable_target_model_size / self.prunable_model_size

        print("********** Initializing L0 Module **********") 
        for pruning_module in self.pruning_modules:
            print(f"***** {pruning_module} *****")
            print(f"z.shape", self.masks[pruning_module].z_loga.shape)
            print(f"size", self.masks[pruning_module].mask_size)
        print(f"prunable model size: {self.prunable_model_size}")
        
    
    def set_model_info(self, cfg, n_matrix_mlp):
        ns = NS() 
        ns.hidden_size = cfg.d_model
        ns.intermediate_size = cfg.intermediate_size
        ns.num_attention_heads = cfg.n_heads
        ns.num_attention_kv_heads = cfg.n_kv_heads
        ns.mlp_num_per_layer = 1
        ns.dim_per_head = ns.hidden_size // ns.num_attention_heads 
        ns.num_layers = cfg.n_layers
        ns.vocab_size = cfg.vocab_size

        ns.params_per_head_layer = ns.hidden_size * ns.dim_per_head * (ns.num_attention_heads + ns.num_attention_kv_heads) * 2
        ns.params_per_head =  ns.params_per_head_layer // ns.num_attention_kv_heads
        ns.params_per_mlp_layer = ns.hidden_size * ns.intermediate_size * n_matrix_mlp
        ns.params_per_intermediate_dim = ns.params_per_mlp_layer // ns.intermediate_size

        ns.full_model_size = (ns.params_per_head_layer + ns.params_per_mlp_layer) * ns.num_layers
        return ns
        
    def calculate_prunable_model_size(self, ns: NS):
        prunable_mlp_size = ns.params_per_mlp_layer * ns.num_layers
        prunable_head_layer_size = ns.params_per_head_layer * ns.num_layers
        prunable_model_size = 0
        if "hidden" in self.pruning_modules:
            return prunable_mlp_size + prunable_head_layer_size
        if "head_layer" in self.pruning_modules or "head" in self.pruning_modules:
            prunable_model_size += prunable_head_layer_size
        if "mlp" in self.pruning_modules or "intermediate" in self.pruning_modules:
            prunable_model_size += prunable_mlp_size
        return prunable_model_size
        
    def initialize_one_module(self, module_name: str):
        func_name = f"initialize_{module_name}"
        try:
            method = getattr(self, func_name)
        except AttributeError:
            raise NotImplementedError("Instance `{}` does not implement `{}`".format(self, func_name))
        method()
            
    def initialize_hidden(self):
        mask_shape = [self.base_model_info.hidden_size]
        num_params_per_mask=self.base_model_info.hidden_size * 4 + self.base_model_info.hidden_size * 4 * 2
        
        target_hidden_sparsity = None; pd=None; target_mask_size=None; 
        if self.target_model_info is not None:
            target_hidden_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
            target_mask_size = self.target_model_info.hidden_size
            pd = {"lambda_1_hidden": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_hidden": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        hidden_mask = Mask(name="hidden",
                           mask_shape=mask_shape,
                           num_params_per_mask=num_params_per_mask,
                           mask_output_shape=[self.base_model_info.hidden_size],
                           target_sparsity=target_hidden_sparsity,
                           target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["hidden"] = hidden_mask

    def initialize_head(self):
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_kv_heads]
        num_params_per_mask = self.base_model_info.params_per_head
        mask_output_shape = [self.base_model_info.num_layers, 1, self.base_model_info.num_attention_kv_heads, 1] 
        
        target_head_sparsity = None; pd = {} ; target_mask_size=None; 
        if self.target_model_info is not None:
            target_head_sparsity = 1 - self.target_model_info.num_attention_kv_heads / self.base_model_info.num_attention_kv_heads
            target_mask_size = self.target_model_info.num_attention_kv_heads
            pd = {"lambda_1_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_head": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        head_mask = Mask(name="head",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_head_sparsity,
                         target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["head"] = head_mask 

    def initialize_qk_head_dim(self): # only campatible when target model info is available
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_kv_heads, self.base_model_info.dim_per_head]
        num_params_per_mask = 2 * self.base_model_info.hidden_size
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.hidden_size] 
        
        target_qk_head_dim_sparsity = None; pd = {} 
        if self.target_model_info is not None:
            target_qk_head_dim_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
            pd = {"lambda_1_qk_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_qk_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        qk_head_dim = Mask(name="qk_head_dim",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_qk_head_dim_sparsity,
                         target_mask_size=self.target_model_info.hidden_size,
                         device=self.device)
        self.masks["qk_head_dim"] = qk_head_dim 
          
          
    def initialize_vo_head_dim(self): # only campatible when target model info is available
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.num_attention_kv_heads, self.base_model_info.dim_per_head]
        num_params_per_mask = 2 * self.base_model_info.hidden_size
        mask_output_shape = [self.base_model_info.num_layers, self.base_model_info.hidden_size] 
        
        target_vo_head_dim_sparsity = None; pd = {} 
        if self.target_model_info is not None:
            target_vo_head_dim_sparsity = 1 - self.target_model_info.hidden_size / self.base_model_info.hidden_size
            pd = {"lambda_1_vo_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_vo_head_dim": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        vo_head_dim = Mask(name="vo_head_dim",
                         mask_shape=mask_shape,
                         num_params_per_mask=num_params_per_mask,
                         mask_output_shape=mask_output_shape,
                         target_sparsity=target_vo_head_dim_sparsity,
                         device=self.device)
        self.masks["vo_head_dim"] = vo_head_dim 
        
    def initialize_head_layer(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask=self.base_model_info.params_per_head *  self.base_model_info.num_attention_kv_heads
        mask_output_shape = [self.base_model_info.num_layers] 
        
        target_head_layer_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_head_layer_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            pd = {"lambda_1_head_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_head_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        head_layer_mask = Mask(name="head_layer",
                              mask_shape=mask_shape,
                               num_params_per_mask=num_params_per_mask,
                               mask_output_shape=mask_output_shape,
                               target_sparsity=target_head_layer_sparsity,
                               target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["head_layer"] = head_layer_mask
        
    def initialize_intermediate(self):
        mask_shape = [self.base_model_info.num_layers, self.base_model_info.intermediate_size]
        num_params_per_mask=self.base_model_info.params_per_intermediate_dim
        mask_output_shape = [self.base_model_info.num_layers, 1, 1, self.base_model_info.intermediate_size] 
        
        target_int_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_int_sparsity = 1 - self.target_model_info.intermediate_size / self.base_model_info.intermediate_size
            target_mask_size = self.target_model_info.intermediate_size
            pd = {"lambda_1_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_intermediate": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        int_mask = Mask(name="intermediate",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_int_sparsity,
                        target_mask_size=target_mask_size,
                           device=self.device,
                           eval_target_model=self.eval_target_model)
        self.masks["intermediate"] = int_mask
       

    def initialize_mlp(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask=self.base_model_info.params_per_mlp_layer
        mask_output_shape = [self.base_model_info.num_layers] 
        
        target_mlp_sparsity = None; pd = {}; target_mask_size=None; 
        if self.target_model_info is not None:
            target_mlp_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            pd = {"lambda_1_mlp": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_mlp": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        mlp_mask = Mask(name="mlp",
                        mask_shape=mask_shape,
                        num_params_per_mask=num_params_per_mask,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_mlp_sparsity,
                        target_mask_size=target_mask_size,
                        eval_target_model=self.eval_target_model,
                        device=self.device)
        self.masks["mlp"] = mlp_mask 

    def initialize_layer(self):
        mask_shape = [self.base_model_info.num_layers]
        num_params_per_mask=self.base_model_info.params_per_head * self.base_model_info.num_attention_kv_heads + self.base_model_info.params_per_mlp_layer
        mask_output_shape = [self.base_model_info.num_layers] 
        
        target_layer_sparsity = None; target_mask_size=None;  pd = {}
        if self.target_model_info is not None:
            target_layer_sparsity = 1 - self.target_model_info.num_layers / self.base_model_info.num_layers
            target_mask_size = self.target_model_info.num_layers
            pd = {"lambda_1_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device)),
                  "lambda_2_layer": torch.nn.Parameter(torch.tensor(0.0, device=self.device))}
            self.lambdas.update(pd)
        
        layer_mask = Mask(name="layer",
                         mask_shape=mask_shape,
                          num_params_per_mask=num_params_per_mask,
                          mask_output_shape=mask_output_shape,
                          target_sparsity=target_layer_sparsity,
                          target_mask_size=target_mask_size,
                          eval_target_model=self.eval_target_model,
                          device=self.device) 
        self.masks["layer"] = layer_mask 
    
    def constrain_parameters(self):
        for key in self.masks:
            self.masks[key].constrain_parameters()

    def calculate_expected_score_sparsity(self):
        expected_scores = {}
        expected_sparsitys = {}
        for key in self.masks:
            score, sparsity = self.masks[key].calculate_expected_score_sparsity()
            expected_scores[key] = score
            expected_sparsitys[key] = sparsity
        return expected_scores, expected_sparsitys
    
    def transform_scores_for_head(self, expected_scores: dict):
        head_score = expected_scores["head"] # 12 * 12

        head_layer_score = None
        if "head_layer" in expected_scores:
            head_layer_score = expected_scores["head_layer"]
        elif "layer" in expected_scores:
            head_layer_score = expected_scores["layer"] # 12
        if head_layer_score is not None:
            head_layer_score = head_layer_score.view(-1, 1) # 12 * 1
       
        return head_layer_score, head_score

    def transform_scores_for_mlp(self, expected_scores: dict):
        mlp_score = None
        if "mlp" in expected_scores:
            mlp_score = expected_scores["mlp"] # 12
        elif "layer" in expected_scores:
            mlp_score = expected_scores["layer"] # 12
        if mlp_score is not None:
            mlp_score = mlp_score.unsqueeze(-1)
        
        intermediate_score = expected_scores["intermediate"] # 12 * 3072
        return mlp_score, intermediate_score


    def get_expected_num_params(self, expected_scores: dict): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 
        # 12 * 12
        head_layer_score, head_score = self.transform_scores_for_head(expected_scores)
        mlp_score, int_score = self.transform_scores_for_mlp(expected_scores)
        
        head_score = (head_layer_score * head_score) # 12 * 12
        int_score = (mlp_score * int_score) # 12 * 3072

        qk_score = None
        if "qk_head_dim" in expected_scores:
            qk_head_dim_score = expected_scores["qk_head_dim"] # num_layer * hidden_size
            vo_head_dim_score = expected_scores["vo_head_dim"] # num_layer * hidden_size
            qk_head_dim_score = qk_head_dim_score.view(qk_head_dim_score.shape[0], -1) # 12 * 768
            vo_head_dim_score = vo_head_dim_score.view(vo_head_dim_score.shape[0], -1) # 12 * 768
            head_score = torch.repeat_interleave(head_score, self.base_model_info.dim_per_head, dim=1) # 12 * 768

            qk_score = head_score * qk_head_dim_score # 12 * 768
            vo_score = head_score * vo_head_dim_score # 12 * 768
                
        if "hidden" in expected_scores:
            hidden_score = expected_scores["hidden"] # 768 
            
            if qk_score is None:
                num_parameters += torch.outer(hidden_score, head_score.reshape(-1)).sum() * self.masks.head.num_params_per_mask / self.base_model_info.hidden_size # 768 * 144
                num_parameters += torch.outer(hidden_score, int_score.reshape(-1)).sum() * self.masks.intermediate.num_params_per_mask / self.base_model_info.hidden_size # 768 * 36864
            else:
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), qk_score.unsqueeze(1))) * 2 # 12 * 768 * 768
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), vo_score.unsqueeze(1))) * 2 # 12 * 768 * 768
                num_parameters += torch.sum(torch.matmul(hidden_score.reshape(1, -1, 1), int_score.unsqueeze(1))) * 3 # 12 * 768 * 3072
        else:
            num_parameters += torch.sum(head_score) * self.masks.head.num_params_per_mask
            num_parameters += torch.sum(int_score) * self.masks.intermediate.num_params_per_mask
        return num_parameters
    
    def get_target_sparsity(self, pruned_steps: int, full_sparsity: float = None):
        target_sparsity = full_sparsity
        if getattr(self, "lagrangian_warmup_steps", 0) > 0:
            target_sparsity = (target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup_steps) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps: int):
        def _lag_loss(expected_sparsity: torch.tensor, target_sparsity: float, lambda_1: torch.tensor, lambda_2: torch.tensor):
            lagrangian_loss = lambda_1 * (expected_sparsity - target_sparsity) + lambda_2 * (expected_sparsity - target_sparsity) ** 2 
            lagrangian_loss = lagrangian_loss.mean()
            return lagrangian_loss

        target_sparsity = self.get_target_sparsity(pruned_steps, self.target_sparsity)            
        expected_scores, expected_sparsitys = self.calculate_expected_score_sparsity()
        expected_size = self.get_expected_num_params(expected_scores) #! calculate \bar s
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        
        return_v = {}
        if self.target_model_info is None:
            lagrangian_loss = _lag_loss(expected_sparsity, target_sparsity, self.lambdas["lambda_1"], self.lambdas["lambda_2"])
            return_v = {"expected_sparsity": expected_sparsity.item(), "target_sparsity": target_sparsity}
            for key in expected_sparsitys:
                return_v[f"expected_{key}_sparsity"] = expected_sparsitys[key].mean().item()
        else:
            lagrangian_loss = 0
            return_v = {}
            for pruning_module in self.pruning_modules:
                ts = self.get_target_sparsity(pruned_steps, self.masks[pruning_module].target_sparsity)
                expected_ts = expected_sparsitys[pruning_module] 
                lagrangian_loss += _lag_loss(expected_ts, ts, self.lambdas[f"lambda_1_{pruning_module}"], self.lambdas[f"lambda_2_{pruning_module}"])
                expected_ts = expected_ts.mean().item()
                return_v.update({"expected_{}_sparsity".format(pruning_module): expected_ts, "target_{}_sparsity".format(pruning_module): ts})
            return_v["expected_sparsity"] = expected_sparsity.item()
            return_v["target_sparsity"] = target_sparsity


        # return_v might not matter
        return lagrangian_loss, return_v
 
    def forward(self, calculate_lagrangian: bool = False, pruned_steps: int = 0):
        self.constrain_parameters()
        if calculate_lagrangian:
            return self.lagrangian_regularization(pruned_steps)
        
        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}
        
        if "layer" in self.pruning_modules:
            zs.pop("layer_z")
            zs["mlp_z"] = []
            zs["head_layer_z"] = []
        
        if self.training:
            for pruning_module in self.pruning_modules:
                mask = self.masks[pruning_module]
                z = mask.sample_z()
                zs[f"{pruning_module}_z"] = z
        else: # removed layerwise! 
            with torch.no_grad():
                for pruning_module in self.pruning_modules:
                    mask = self.masks[pruning_module]
                    z = mask.deterministic_z()
                    zs[f"{pruning_module}_z"] = z
        if "layer_z" in zs:
            zs["mlp_z"] = zs.pop("layer_z")
            zs["head_layer_z"] = zs["mlp_z"]
        return zs 
