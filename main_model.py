"""
Core CSDI model implementation with base class and dataset-specific subclasses.

Based on "CSDI: Conditional Score-based Diffusion Models for Probabilistic
Time Series Imputation" by Tashiro et al. (NeurIPS 2021)
"""

import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    """Base class for CSDI (Conditional Score-based Diffusion) models.
    
    This class implements the core diffusion model for time series imputation,
    including the forward diffusion process and reverse denoising process.
    """

    def __init__(self, target_dim, config, device):
        """Initialize the CSDI base model.
        
        Args:
            target_dim: Number of features/dimensions in the time series
            config: Configuration dictionary with model and diffusion parameters
            device: PyTorch device for computation
        """
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        """Generate positional embedding for time steps.
        
        Args:
            pos: Position tensor of shape (B, L)
            d_model: Embedding dimension
            
        Returns:
            Positional embedding tensor of shape (B, L, d_model)
        """
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        """Generate random mask for training with random missing pattern.
        
        Args:
            observed_mask: Binary mask indicating observed values
            
        Returns:
            Conditional mask for training
        """
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        """Generate historical pattern mask for training.
        
        Args:
            observed_mask: Binary mask indicating observed values
            for_pattern_mask: Optional pattern mask from another sample
            
        Returns:
            Conditional mask based on historical patterns
        """
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        """Generate test pattern mask.
        
        Args:
            observed_mask: Binary mask indicating observed values
            test_pattern_mask: Pattern mask for testing
            
        Returns:
            Combined mask for testing
        """
        return observed_mask * test_pattern_mask

    def get_side_info(self, observed_tp, cond_mask):
        """Get side information (time and feature embeddings) for the model.
        
        Args:
            observed_tp: Observed time points
            cond_mask: Conditional mask
            
        Returns:
            Side information tensor
        """
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        """Calculate validation loss by averaging over all diffusion steps.
        
        Args:
            observed_data: Original observed data
            cond_mask: Conditional mask
            observed_mask: Observation mask
            side_info: Side information tensor
            is_train: Training flag
            
        Returns:
            Average loss over all diffusion steps
        """
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        """Calculate training loss for a specific or random diffusion step.
        
        Args:
            observed_data: Original observed data
            cond_mask: Conditional mask
            observed_mask: Observation mask
            side_info: Side information tensor
            is_train: Training flag (1 for training, 0 for validation)
            set_t: Specific time step (-1 for random)
            
        Returns:
            Loss value
        """
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        """Prepare input for the diffusion model.
        
        Args:
            noisy_data: Noised data from forward diffusion
            observed_data: Original observed data
            cond_mask: Conditional mask
            
        Returns:
            Input tensor for the diffusion model
        """
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        """Perform imputation using reverse diffusion process.
        
        Args:
            observed_data: Original observed data
            cond_mask: Conditional mask indicating observed values
            side_info: Side information tensor
            n_samples: Number of imputation samples to generate
            
        Returns:
            Imputed samples tensor of shape (B, n_samples, K, L)
        """
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        """Forward pass for training or validation.
        
        Args:
            batch: Input batch dictionary
            is_train: Training flag (1 for training, 0 for validation)
            
        Returns:
            Loss value
        """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        """Evaluate the model by generating imputation samples.
        
        Args:
            batch: Input batch dictionary
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (samples, observed_data, target_mask, observed_mask, observed_tp)
        """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Physio(CSDI_base):
    """CSDI model specialized for healthcare/physio dataset.
    
    Inherits from CSDI_base and implements data processing specific
    to the PhysioNet Challenge 2012 healthcare dataset.
    """

    def __init__(self, config, device, target_dim=35):
        """Initialize the CSDI model for physio data.
        
        Args:
            config: Configuration dictionary
            device: PyTorch device
            target_dim: Number of features (35 for physio dataset)
        """
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        """Process batch data for the physio dataset.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Tuple of processed tensors
        """
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_PM25(CSDI_base):
    """CSDI model specialized for PM2.5 air quality dataset.
    
    Inherits from CSDI_base and implements data processing specific
    to the PM2.5 air quality dataset.
    """

    def __init__(self, config, device, target_dim=36):
        """Initialize the CSDI model for PM2.5 data.
        
        Args:
            config: Configuration dictionary
            device: PyTorch device
            target_dim: Number of features (36 for PM2.5 dataset)
        """
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        """Process batch data for the PM2.5 dataset.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Tuple of processed tensors
        """
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Forecasting(CSDI_base):
    """CSDI model specialized for time series forecasting.
    
    Inherits from CSDI_base and implements additional features
    for forecasting tasks including feature sampling.
    """

    def __init__(self, config, device, target_dim):
        """Initialize the CSDI model for forecasting.
        
        Args:
            config: Configuration dictionary
            device: PyTorch device
            target_dim: Number of features
        """
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        """Process batch data for forecasting.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Tuple of processed tensors including feature_id
        """
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id = torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0], -1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id,
        )

    def sample_features(self, observed_data, observed_mask, feature_id, gt_mask):
        """Sample a subset of features for training efficiency.
        
        Args:
            observed_data: Original observed data
            observed_mask: Observation mask
            feature_id: Feature indices
            gt_mask: Ground truth mask
            
        Returns:
            Tuple of sampled tensors
        """
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []

        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k, ind[:size]])
            extracted_mask.append(observed_mask[k, ind[:size]])
            extracted_feature_id.append(feature_id[k, ind[:size]])
            extracted_gt_mask.append(gt_mask[k, ind[:size]])
        extracted_data = torch.stack(extracted_data, 0)
        extracted_mask = torch.stack(extracted_mask, 0)
        extracted_feature_id = torch.stack(extracted_feature_id, 0)
        extracted_gt_mask = torch.stack(extracted_gt_mask, 0)
        return extracted_data, extracted_mask, extracted_feature_id, extracted_gt_mask

    def get_side_info(self, observed_tp, cond_mask, feature_id=None):
        """Get side information for forecasting model.
        
        Args:
            observed_tp: Observed time points
            cond_mask: Conditional mask
            feature_id: Optional feature indices for sampled features
            
        Returns:
            Side information tensor
        """
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        """Forward pass for forecasting training or validation.
        
        Args:
            batch: Input batch dictionary
            is_train: Training flag (1 for training, 0 for validation)
            
        Returns:
            Loss value
        """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask, feature_id, gt_mask = \
                self.sample_features(observed_data, observed_mask, feature_id, gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else:  # test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        """Evaluate the forecasting model.
        
        Args:
            batch: Input batch dictionary
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (samples, observed_data, target_mask, observed_mask, observed_tp)
        """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1 - gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
