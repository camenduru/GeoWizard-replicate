import os
from cog import BasePredictor, Input, Path
from typing import List
import sys
sys.path.append('/content/geowizard-hf')
os.chdir('/content/geowizard-hf')

from PIL import Image
import torch
from tqdm.auto import tqdm
from models.depth_normal_pipeline_clip import DepthNormalEstimationPipeline
from utils.seed_all import seed_all
from utils.depth2normal import *
from diffusers import DDIMScheduler, AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

def depth_normal(img,
                denoising_steps,
                ensemble_size,
                processing_res,
                seed,
                domain,
                pipe):

    seed = int(seed)
    if seed >= 0:
        torch.manual_seed(seed)
    pipe_out = pipe(
        img,
        denoising_steps=denoising_steps,
        ensemble_size=ensemble_size,
        processing_res=processing_res,
        batch_size=0,
        domain=domain,
        #seed = seed,
        show_progress_bar=True,
    )
    depth_colored = pipe_out.depth_colored
    normal_colored = pipe_out.normal_colored
    return depth_colored, normal_colored

class Predictor(BasePredictor):
    def setup(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        stable_diffusion_repo_path = "stabilityai/stable-diffusion-2-1-unclip"
        vae = AutoencoderKL.from_pretrained(stable_diffusion_repo_path, subfolder='vae')
        scheduler = DDIMScheduler.from_pretrained(stable_diffusion_repo_path, subfolder='scheduler')
        sd_image_variations_diffusers_path = 'lambdalabs/sd-image-variations-diffusers'
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(sd_image_variations_diffusers_path, subfolder="image_encoder")
        feature_extractor = CLIPImageProcessor.from_pretrained(sd_image_variations_diffusers_path, subfolder="feature_extractor")
        unet = UNet2DConditionModel.from_pretrained('.', subfolder="unet")
        self.pipe = DepthNormalEstimationPipeline(vae=vae, image_encoder=image_encoder, feature_extractor=feature_extractor, unet=unet, scheduler=scheduler)
        try:
            import xformers
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
        self.pipe = self.pipe.to(device)
    def predict(
        self,
        input_image: Path = Input(description="Input image"),
        denoising_steps: int = Input(default=10, ge=1, le=50),
        ensemble_size: int = Input(default=4, ge=1, le=15),
        processing_res: int = Input(choices=[0, 768], default=768),
        seed: int = Input(default=123),
        domain: str = Input(choices=['outdoor', 'indoor', 'object'], default="indoor"),
    ) -> List[Path]:
        depth, normal = depth_normal(input_image, denoising_steps, ensemble_size, processing_res, seed, domain, self.pipe)
        return [Path(depth), Path(normal)]