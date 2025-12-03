import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from PIL import Image

from contrast_policies.kde_contrast_decoding import gaussian_kernel_jax, kde_jax, scott_rule_jax


class DistributionVisualizer:
    def __init__(self, policy, bandwidth_factor=1.0, smooth_kernel_size=11):
        self.policy = policy
        self.bandwidth_factor = bandwidth_factor
        self.smooth_kernel_size = smooth_kernel_size
    
    def visualize(self, image, save_path, actions=None, logits=None):
        if actions is not None:
            actions, probs = self._octo_actions_to_distribution(actions)
        elif logits is not None:
            actions, probs = self._openvla_logits_to_distribution(logits)
        else:
            raise ValueError("Either actions or logits must be provided.")
        
        xx_probs, yy_probs, xx_actions, yy_actions = self._meshgrid_actions(actions, probs)

        save_dir = "/".join(save_path.split("/")[:-1])
        os.makedirs(save_dir, exist_ok=True)
        Image.fromarray(image).save(save_path)
        
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        xxyy_probs = xx_probs * yy_probs
        # xxyy_probs[xxyy_probs < 1e-8] = None
        ax.plot_surface(xx_actions, yy_actions, xxyy_probs, cmap='jet')
        ax.view_init(elev=75, azim=90)
        # invert y axis
        ax.invert_yaxis()
        # hide z ticks and line
        ax.set_zticks([])
        # hide mesh
        ax.grid(None)
        # hide background
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # hide background line
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # set ticklabel [-0.2, -0.1, 0.0, 0.1, 0.2]
        ax.set_xticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
        plt.savefig(save_path.replace(".", "_distribution."), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
   
    def _meshgrid_actions(self, actions, probs):
        # actions: (7, N), probs: (7, N)
        # only take x, y
        x_actions, y_actions = actions[:2]
        x_probs, y_probs = probs[:2]
        x_probs, y_probs = self._smooth_probs(x_probs), self._smooth_probs(y_probs)
        yy_probs, xx_probs = np.meshgrid(y_probs, x_probs)
        yy_actions, xx_actions = np.meshgrid(y_actions, x_actions)
        return xx_probs, yy_probs, xx_actions, yy_actions
        
    def _octo_actions_to_distribution(self, actions):
        # only consider the first action
        actions = actions[:, 0]
        # (B, D) -> (D, B)
        actions = actions.transpose(1, 0)
        # (D,), (D,)
        # min_actions, max_actions = actions.min(axis=-1), actions.max(axis=-1)
        # max_actions = jnp.abs(actions).max(axis=-1)
        max_actions = jnp.ones_like(actions[:, 0]) * 2.0
        min_actions = -max_actions
        # uniform sample (D,) -> (D, 255)
        sample_actions = jnp.linspace(min_actions, max_actions, 255, axis=1)
        
        unnormalized_sample_actions = sample_actions * self.policy.action_std[:, None] + self.policy.action_mean[:, None]
        
        bandwidth = self.bandwidth_factor * scott_rule_jax(actions)
        probs = kde_jax(sample_actions, actions, bandwidth, gaussian_kernel_jax)
        
        return np.array(unnormalized_sample_actions), np.array(probs)
    
    def _openvla_logits_to_distribution(self, logits):
        # reset action token ids
        # [0, 1, 2, ..., 32063] -> [32000, 31999, ..., 0, ..., -63] -> [31999, -64]
        action_token_ids = torch.arange(logits.size(1))
        discretized_actions = self.policy.vla.vocab_size - action_token_ids.cpu()
        discretized_actions -= 1
        
        # choose action tokens
        # [F, F, ..., T, ..., F]
        keep_mask = torch.logical_and(discretized_actions >= 0, discretized_actions <= self.policy.vla.bin_centers.shape[0] - 1).bool()

        discretized_actions = discretized_actions[keep_mask]
        probs = (logits[:, keep_mask] / self.bandwidth_factor).softmax(dim=-1).detach().cpu().numpy()
        num_bins = discretized_actions.shape[0]
        num_actions = logits.shape[0]

        # get normalized action values of each token
        # (bins,) -> (actions, bins)
        normalized_actions = self.policy.vla.bin_centers[discretized_actions]
        normalized_actions = np.repeat(normalized_actions[None], num_actions, axis=0)

        # unnormalize action values of each token
        action_norm_stats = self.policy.get_action_stats(self.policy.unnorm_key)
        mask = np.array(action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool)))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        # (actions,) -> (actions, bins)
        mask = np.stack([mask for _ in range(num_bins)], axis=1)
        action_high = np.stack([action_high for _ in range(num_bins)], axis=1)
        action_low = np.stack([action_low for _ in range(num_bins)], axis=1)
        unnormalized_actions = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)

        return unnormalized_actions, probs

    def _smooth_probs(self, probs):
        if self.smooth_kernel_size == 1:
            return probs
        return np.convolve(probs, np.ones(self.smooth_kernel_size) / self.smooth_kernel_size, mode='same')