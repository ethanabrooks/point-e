# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from PIL import Image
import torch
from tqdm.auto import tqdm
import time

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("creating base model...")
base_name = "base40M"  # use base300M or base1B for better results
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print("creating upsample model...")
upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])

print("downloading base checkpoint...")
base_model.load_state_dict(load_checkpoint(base_name, device))

print("downloading upsampler checkpoint...")
upsampler_model.load_state_dict(load_checkpoint("upsample", device))

# %%
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=["R", "G", "B"],
    guidance_scale=[3.0, 3.0],
)

# %%
from io import BytesIO
import requests
import openai
import os

openai.api_key = os.getenv("OPENAI_KEY")

response = openai.Image.create(
    prompt="a 3D render of a lunar rover, white background", n=1, size="512x512"
)
image_url = response["data"][0]["url"]

response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# %%
t0 = time.time()

# Produce a sample from the model.
samples = None
for x in tqdm(
    sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))
):
    samples = x

print(time.time() - t0)

# %%
pc = sampler.output_to_point_clouds(samples)[0]
fig = plot_point_cloud(
    pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75))
)

# %%
