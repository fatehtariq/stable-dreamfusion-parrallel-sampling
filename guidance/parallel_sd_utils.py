from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline, \
    DDPMParallelScheduler, StableDiffusionParadigmsPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd
from .perpneg_utils import weighted_perpendicular_aggregator


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionParadigmsPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
        #pipe.enable_xformers_memory_efficient_attention()

        # Create wrapped unet pipeline for parallel sampling with multiple GPUs
        ngpu, batch_per_device = torch.cuda.device_count(), 5
        pipe.wrapped_unet = torch.nn.DataParallel(pipe.unet, device_ids=[d for d in range(ngpu)])
        self.wrapped_unet = pipe.wrapped_unet

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDPMParallelScheduler.from_pretrained(model_key, subfolder="scheduler",
                                                               torch_dtype=self.precision_t,
                                                               timestep_spacing="trailing")

        self.parallel = ngpu * batch_per_device

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] Loaded Stable Diffusion with Paradigms scheduler and pipeline...')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        # prompt: [str]

        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def train_step(
            self,
            pred_rgb,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            parallel: int = 10,
            tolerance: float = 0.1,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            full_return: bool = False):
        print("[INFO] Setting up parallel pipeline.", flush=True)

        # 0. Default height and width to unet
        height = height or 512
        width = width or 512

        print("[INFO] Setting up Height and Width of Image Data.", flush=True)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        print("[INFO] Setting up Call Parameters.", flush=True)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        print("[INFO] Extracted Text Embeddings successfully.", flush=True)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        scheduler = self.scheduler

        print("[INFO] Preparing Time steps for Scheduler.", flush=True)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)

        print("[INFO] Producing Latent Variables.", flush=True)

        # 6. Prepare extra step kwargs.
        #extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        # print(scheduler.timesteps)
        stats_pass_count = 0
        stats_flop_count = 0
        parallel = min(parallel, len(scheduler.timesteps))

        begin_idx = 0
        end_idx = parallel
        latents_time_evolution_buffer = torch.stack([latents] * (len(scheduler.timesteps) + 1))

        # We specify the error tolerance as a ratio of the scheduler's noise magnitude. We similarly compute the error tolerance
        # outside of the denoising loop to avoid recomputing it at every step.
        # We will be dividing the norm of the noise, so we store its inverse here to avoid a division at every step.
        noise_array = torch.zeros_like(latents_time_evolution_buffer)
        for j in range(len(scheduler.timesteps)):
            base_noise = torch.randn_like(latents)
            noise = (self.scheduler._get_variance(scheduler.timesteps[j]) ** 0.5) * base_noise
            noise_array[j] = noise.clone()

        print("[INFO] Noise Array created before hand.", flush=True)

        # We specify the error tolerance as a ratio of the scheduler's noise magnitude. We similarly compute the error tolerance
        # outside of the denoising loop to avoid recomputing it at every step.
        # We will be dividing the norm of the noise, so we store its inverse here to avoid a division at every step.
        inverse_variance_norm = 1. / torch.tensor(
            [scheduler._get_variance(scheduler.timesteps[j]) for j in range(len(scheduler.timesteps))] + [0]).to(
            noise_array.device)
        latent_dim = noise_array[0, 0].numel()
        inverse_variance_norm = inverse_variance_norm[:, None] / latent_dim

        scaled_tolerance = (tolerance ** 2)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        cumulative_noise = 0

        start.record()

        while begin_idx < len(scheduler.timesteps):
            # these have shape (parallel_dim, 2*batch_size, ...)
            # parallel_len is at most parallel, but could be less if we are at the end of the timesteps
            # we are processing batch window of timesteps spanning [begin_idx, end_idx)
            parallel_len = end_idx - begin_idx

            block_prompt_embeds = torch.stack([prompt_embeds] * parallel_len)
            block_latents = latents_time_evolution_buffer[begin_idx:end_idx]
            block_t = scheduler.timesteps[begin_idx:end_idx, None].repeat(1, batch_size * num_images_per_prompt)
            t_vec = block_t
            if do_classifier_free_guidance:
                t_vec = t_vec.repeat(1, 2)

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([block_latents] * 2, dim=1) if do_classifier_free_guidance else block_latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_vec)

            # if parallel_len is small, no need to use multiple GPUs
            net = self.wrapped_unet if parallel_len > 3 else self.unet
            # predict the noise residual
            model_output = net(
                latent_model_input.flatten(0, 1),
                t_vec.flatten(0, 1),
                encoder_hidden_states=block_prompt_embeds.flatten(0, 1),
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            per_latent_shape = model_output.shape[1:]
            if do_classifier_free_guidance:
                model_output = model_output.reshape(
                    parallel_len, 2, batch_size * num_images_per_prompt, *per_latent_shape
                )
                noise_pred_uncond, noise_pred_text = model_output[:, 0], model_output[:, 1]
                model_output = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            model_output = model_output.reshape(
                parallel_len * batch_size * num_images_per_prompt, *per_latent_shape
            )

            block_latents_denoise = scheduler.batch_step_no_noise(
                model_output=model_output,
                timesteps=block_t.flatten(0, 1),
                sample=block_latents.flatten(0, 1),
            ).reshape(block_latents.shape)

            # back to shape (parallel_dim, batch_size, ...)
            # now we want to add the pre-sampled noise
            # parallel sampling algorithm requires computing the cumulative drift from the beginning
            # of the window, so we need to compute cumulative sum of the deltas and the pre-sampled noises.
            delta = block_latents_denoise - block_latents
            cumulative_delta = torch.cumsum(delta, dim=0)
            cumulative_noise = torch.cumsum(noise_array[begin_idx:end_idx], dim=0)

            # if we are using an ODE-like scheduler (like DDIM), we don't want to add noise
            if scheduler._is_ode_scheduler:
                cumulative_noise = 0

            block_latents_new = latents_time_evolution_buffer[begin_idx][None,] + cumulative_delta + cumulative_noise
            cur_error_vec = (block_latents_new - latents_time_evolution_buffer[begin_idx + 1:end_idx + 1]).reshape(
                parallel_len, batch_size * num_images_per_prompt, -1)
            cur_error = torch.linalg.norm(cur_error_vec, dim=-1).pow(2)
            error_ratio = cur_error * inverse_variance_norm[begin_idx + 1:end_idx + 1]

            # find the first index of the vector error_ratio that is greater than error tolerance
            # we can shift the window for the next iteration up to this index
            error_ratio = torch.nn.functional.pad(
                error_ratio, (0, 0, 0, 1), value=1e9
            )  # handle the case when everything is below ratio, by padding the end of parallel_len dimension
            any_error_at_time = torch.max(error_ratio > scaled_tolerance, dim=1).values.int()
            ind = torch.argmax(any_error_at_time).item()

            # compute the new begin and end idxs for the window
            new_begin_idx = begin_idx + min(1 + ind, parallel)
            new_end_idx = min(new_begin_idx + parallel, len(scheduler.timesteps))

            # store the computed latents for the current window in the global buffer
            latents_time_evolution_buffer[begin_idx + 1:end_idx + 1] = block_latents_new
            # initialize the new sliding window latents with the end of the current window,
            # should be better than random initialization
            latents_time_evolution_buffer[end_idx:new_end_idx + 1] = latents_time_evolution_buffer[end_idx][None,]

            begin_idx = new_begin_idx
            end_idx = new_end_idx

            stats_pass_count += 1
            stats_flop_count += parallel_len

        latents = latents_time_evolution_buffer[-1]

        print("pass count", stats_pass_count)
        print("flop count", stats_flop_count)

        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)
        grad_scale = 1
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (cumulative_noise - torch.randn_like(latents))
        grad = torch.nan_to_num(grad)
        targets = (latents - grad).detach()

        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        print(start.elapsed_time(end))
        print("done", flush=True)

        print("[INFO] Image Generation complete, returning loss...", flush=True)

        return loss

    def train_step_perpneg(self, text_embeddings, weights, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1,
                           save_guidance_path: Path = None):

        B = pred_rgb.shape[0]
        K = (text_embeddings.shape[0] // B) - 1  # maximum number of prompts

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * (1 + K))
            tt = torch.cat([t] * (1 + K))
            unet_output = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = unet_output[:B], unet_output[B:]
            delta_noise_preds = noise_pred_text - noise_pred_uncond.repeat(K, 1, 1, 1)
            noise_pred = noise_pred_uncond + guidance_scale * weighted_perpendicular_aggregator(delta_noise_preds,
                                                                                                weights, B)

            # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_512 = self.decode_latents(latents)

                # visualize predicted denoised image
                # The following block of code is equivalent to `predict_start_from_noise`...
                # see zero123_utils.py's version for a simpler implementation.
                alphas = self.scheduler.alphas.to(latents)
                total_timesteps = self.max_step - self.min_step + 1
                index = total_timesteps - t.to(latents.device) - 1
                b = len(noise_pred)
                a_t = alphas[index].reshape(b, 1, 1, 1).to(self.device)
                sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
                sqrt_one_minus_at = sqrt_one_minus_alphas[index].reshape((b, 1, 1, 1)).to(self.device)
                pred_x0 = (latents_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()  # current prediction for x_0
                result_hopefully_less_noisy_image = self.decode_latents(pred_x0.to(latents.type(self.precision_t)))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy.to(pred_x0).type(self.precision_t))

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_512, result_noisier_image, result_hopefully_less_noisy_image], dim=0)
                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5,
                        latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                                  device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50,
                      guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts)  # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'],
                        help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()




