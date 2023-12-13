from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.notebooks import decode_latent_mesh

from dataclasses import dataclass, field
from typing import List
import os

import numpy as np
import threestudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *


@threestudio.register("shap-e-guidance")
class shap_e_Guidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        model_name: str = "transmitter"
        vae_name: str = "text300M"
        finetune_model_name: str = "finetune_model_name.pt"
        diff_config_name: str = "diffusion"
        guidance_scale: float = 15.0
        skip: int = 4
        cache_dir: str = "custom/threestudio-shap-e/shap-e/cache"

    cfg: Config

    def configure(self) -> None:
        pass
    
    def densify(self, factor=2):
        pass

    def __call__(
        self,
        prompt,
        **kwargs,
    ):
        
        threestudio.info(f"Loading shap-e guidance ...")
        device = self.device 
        xm = load_model(self.cfg.model_name, device=device, cache_dir=self.cfg.cache_dir)
        model = load_model(self.cfg.vae_name, device=device, cache_dir=self.cfg.cache_dir)
        if os.path.exists(os.path.join(self.cfg.cache_dir, self.cfg.finetune_model_name)):
            model.load_state_dict(torch.load(os.path.join(self.cfg.cache_dir, self.cfg.finetune_model_name), map_location=device)['model_state_dict'])
        diffusion = diffusion_from_config_shape(load_config('diffusion', cache_dir=self.cfg.cache_dir))
        threestudio.info(f"Loaded shap-e guidance!")

        batch_size = 1
        guidance_scale = self.cfg.guidance_scale
        prompt = str(prompt)

        latents = sample_latents(
            batch_size=batch_size,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt] * batch_size),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        # render_mode = 'nerf' # you can change this to 'stf'
        # size = 256 # this is the size of the renders; higher values take longer to render.
        # cameras = create_pan_cameras(size, device)
        # self.shapeimages = decode_latent_images(xm, latents[0], cameras, rendering_mode=render_mode)
        pc = decode_latent_mesh(xm, latents[0]).tri_mesh()

        skip = self.cfg.skip
        coords = pc.verts
        rgb = np.concatenate([pc.vertex_channels['R'][:,None],pc.vertex_channels['G'][:,None],pc.vertex_channels['B'][:,None]],axis=1) 

        coords = coords[::skip]
        rgb = rgb[::skip]

        return coords,rgb
