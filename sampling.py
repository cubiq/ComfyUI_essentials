import comfy.samplers
import comfy.sample
import torch
from nodes import common_ksampler
from .utils import expand_mask

class KSamplerVariationsWithNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "latent_image": ("LATENT", ),
                    "main_seed": ("INT:seed", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "variation_strength": ("FLOAT", {"default": 0.17, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                    #"start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    #"end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    #"return_with_leftover_noise": (["disable", "enable"], ),
                    "variation_seed": ("INT:seed", {"default": 12345, "min": 0, "max": 0xffffffffffffffff}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    # From https://github.com/BlenderNeko/ComfyUI_Noise/
    def slerp(self, val, low, high):
        dims = low.shape

        low = low.reshape(dims[0], -1)
        high = high.reshape(dims[0], -1)

        low_norm = low/torch.norm(low, dim=1, keepdim=True)
        high_norm = high/torch.norm(high, dim=1, keepdim=True)

        low_norm[low_norm != low_norm] = 0.0
        high_norm[high_norm != high_norm] = 0.0

        omega = torch.acos((low_norm*high_norm).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high

        return res.reshape(dims)

    def prepare_mask(self, mask, shape):
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
        mask = mask.expand((-1,shape[1],-1,-1))
        if mask.shape[0] < shape[0]:
            mask = mask.repeat((shape[0] -1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        return mask

    def execute(self, model, latent_image, main_seed, steps, cfg, sampler_name, scheduler, positive, negative, variation_strength, variation_seed, denoise):
        if main_seed == variation_seed:
            variation_seed += 1

        end_at_step = steps #min(steps, end_at_step)
        start_at_step = round(end_at_step - end_at_step * denoise)

        force_full_denoise = True
        disable_noise = True

        device = comfy.model_management.get_torch_device()

        # Generate base noise
        batch_size, _, height, width = latent_image["samples"].shape
        generator = torch.manual_seed(main_seed)
        base_noise = torch.randn((1, 4, height, width), dtype=torch.float32, device="cpu", generator=generator).repeat(batch_size, 1, 1, 1).cpu()

        # Generate variation noise
        generator = torch.manual_seed(variation_seed)
        variation_noise = torch.randn((batch_size, 4, height, width), dtype=torch.float32, device="cpu", generator=generator).cpu()

        slerp_noise = self.slerp(variation_strength, base_noise, variation_noise)

        # Calculate sigma
        comfy.model_management.load_model_gpu(model)
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        sigma = sigma.detach().cpu().item()

        work_latent = latent_image.copy()
        work_latent["samples"] = latent_image["samples"].clone() + slerp_noise * sigma

        # if there's a mask we need to expand it to avoid artifacts, 5 pixels should be enough
        if "noise_mask" in latent_image:
            noise_mask = self.prepare_mask(latent_image["noise_mask"], latent_image['samples'].shape)
            work_latent["samples"] = noise_mask * work_latent["samples"] + (1-noise_mask) * latent_image["samples"]
            work_latent['noise_mask'] = expand_mask(latent_image["noise_mask"].clone(), 5, True)

        return common_ksampler(model, main_seed, steps, cfg, sampler_name, scheduler, positive, negative, work_latent, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


class KSamplerVariationsStochastic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                    "model": ("MODEL",),
                    "latent_image": ("LATENT", ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "variation_seed": ("INT:seed", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "variation_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step":0.05, "round": 0.01}),
                    #"variation_sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                    "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.05, "round": 0.01}),
                }}

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, model, latent_image, noise_seed, steps, cfg, sampler, scheduler, positive, negative, variation_seed, variation_strength, cfg_scale, variation_sampler="dpmpp_2m_sde"):
        # Stage 1: composition sampler
        force_full_denoise = False # return with leftover noise = "enable"
        disable_noise = False # add noise = "enable"

        end_at_step = max(int(steps * (1-variation_strength)), 1)
        start_at_step = 0

        work_latent = latent_image.copy()
        batch_size = work_latent["samples"].shape[0]
        work_latent["samples"] = work_latent["samples"][0].unsqueeze(0)

        stage1 = common_ksampler(model, noise_seed, steps, cfg, sampler, scheduler, positive, negative, work_latent, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]

        if batch_size > 1:
            stage1["samples"] = stage1["samples"].clone().repeat(batch_size, 1, 1, 1)

        # Stage 2: variation sampler
        force_full_denoise = True
        disable_noise = True
        cfg = max(cfg * cfg_scale, 1.0)
        start_at_step = end_at_step
        end_at_step = steps

        return common_ksampler(model, variation_seed, steps, cfg, variation_sampler, scheduler, positive, negative, stage1, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)


SAMPLING_CLASS_MAPPINGS = {
    "KSamplerVariationsStochastic+": KSamplerVariationsStochastic,
    "KSamplerVariationsWithNoise+": KSamplerVariationsWithNoise,
}

SAMPLING_NAME_MAPPINGS = {
    "KSamplerVariationsStochastic+": "🔧 KSampler Stochastic Variations",
    "KSamplerVariationsWithNoise+": "🔧 KSampler Variations with Noise Injection",
}