import os
import comfy.samplers
import comfy.sample
import torch
from nodes import common_ksampler
from .utils import expand_mask, FONTS_DIR, parse_string_to_list
import torchvision.transforms.v2 as T
import torch.nn.functional as F

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

class InjectLatentNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "latent": ("LATENT", ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "noise_strength": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step":0.01, "round": 0.01}),
                }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, latent, noise_seed, noise_strength):
        torch.manual_seed(noise_seed)
        noise_latent = latent.copy()
        noise_latent["samples"] = noise_latent["samples"].clone() + torch.randn_like(noise_latent["samples"]) * noise_strength

        return (noise_latent, )

class FluxSamplerParams:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "conditioning": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),

                    "noise": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "?" }),
                    "sampler": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "ipndm" }),
                    "scheduler": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "simple" }),
                    "steps": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "20" }),
                    "guidance": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "3.5" }),
                    "max_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "1.15" }),
                    "base_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "0.5" }),
                    "split_sigmas": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "1.0" }),
                    "denoise": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "1.0" }),
                }}
    
    RETURN_TYPES = ("LATENT","SAMPLER_PARAMS")
    RETURN_NAMES = ("latent", "params")
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, model, conditioning, latent_image, noise, sampler, scheduler, steps, guidance, max_shift, base_shift, split_sigmas, denoise):
        import random
        import time
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced, SplitSigmasDenoise
        from comfy_extras.nodes_latent import LatentBatch
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux
        from node_helpers import conditioning_set_values

        noise = noise.replace("\n", ",").split(",")
        noise = [random.randint(0, 999999) if "?" in n else int(n) for n in noise]
        if not noise:
            noise = [random.randint(0, 999999)]

        if sampler == '*':
            sampler = comfy.samplers.KSampler.SAMPLERS
        elif sampler.startswith("!"):
            sampler = sampler.replace("\n", ",").split(",")
            sampler = [s.strip("! ") for s in sampler]
            sampler = [s for s in comfy.samplers.KSampler.SAMPLERS if s not in sampler]
        else:
            sampler = sampler.replace("\n", ",").split(",")
            sampler = [s.strip() for s in sampler if s.strip() in comfy.samplers.KSampler.SAMPLERS]
        if not sampler:
            sampler = ['ipndm']

        if scheduler == '*':
            scheduler = comfy.samplers.KSampler.SCHEDULERS
        elif scheduler.startswith("!"):
            scheduler = scheduler.replace("\n", ",").split(",")
            scheduler = [s.strip("! ") for s in scheduler]
            scheduler = [s for s in comfy.samplers.KSampler.SCHEDULERS if s not in scheduler]
        else:
            scheduler = scheduler.replace("\n", ",").split(",")
            scheduler = [s.strip() for s in scheduler]
            scheduler = [s for s in scheduler if s in comfy.samplers.KSampler.SCHEDULERS]
        if not scheduler:
            scheduler = ['simple']

        steps = steps.replace("\n", ",").split(",")
        steps = [int(s) for s in steps]
        if not steps:
            steps = [20]
        
        denoise = parse_string_to_list(denoise)
        if not denoise:
            denoise = [1.0]

        guidance = parse_string_to_list(guidance)
        if not guidance:
            guidance = [3.5]
        
        max_shift = parse_string_to_list(max_shift)
        if not max_shift:
            max_shift = [1.15]
        
        base_shift = parse_string_to_list(base_shift)
        if not base_shift:
            base_shift = [0.5]
        
        split_sigmas = parse_string_to_list(split_sigmas)
        if not split_sigmas:
            split_sigmas = [1.0]

        out_latent = None
        out_params = []

        basicschedueler = BasicScheduler()
        basicguider = BasicGuider()
        samplercustomadvanced = SamplerCustomAdvanced()
        latentbatch = LatentBatch()
        modelsamplingflux = ModelSamplingFlux()
        splitsigmadenoise = SplitSigmasDenoise()
        width = latent_image["samples"].shape[3]*8
        height = latent_image["samples"].shape[2]*8

        for n in noise:
            randnoise = Noise_RandomNoise(n)
            for ms in max_shift:
                for bs in base_shift:
                    work_model = modelsamplingflux.patch(model, ms, bs, width, height)[0]
                    for g in guidance:
                        cond = conditioning_set_values(conditioning, {"guidance": g})
                        guider = basicguider.get_guider(work_model, cond)[0]
                        for s in sampler:
                            samplerobj = comfy.samplers.sampler_object(s)
                            for sc in scheduler:
                                for st in steps:
                                    for d in denoise:
                                        sigmas = basicschedueler.get_sigmas(work_model, sc, st, d)[0]
                                        for ss in split_sigmas:
                                            sigmas = splitsigmadenoise.get_sigmas(sigmas, ss)[1]
                                            start_time = time.time()
                                            latent = samplercustomadvanced.sample(randnoise, guider, samplerobj, sigmas, latent_image)[1]
                                            elapsed_time = time.time() - start_time
                                            out_params.append({"time": elapsed_time,
                                                               "seed": n,
                                                               "sampler": s,
                                                               "scheduler": sc,
                                                               "steps": st,
                                                               "guidance": g,
                                                               "max_shift": ms,
                                                               "base_shift": bs,
                                                               "denoise": d,
                                                               "split_sigmas": ss})

                                            if out_latent is None:
                                                out_latent = latent
                                            else:
                                                out_latent = latentbatch.batch(out_latent, latent)[0]

        return (out_latent, out_params)

class PlotParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                    "params": ("SAMPLER_PARAMS", ),
                    "order_by": (["none", "time", "seed", "steps", "denoise", "sampler", "scheduler"], ),
                    "cols_value": (["none", "time", "seed", "steps", "denoise", "sampler", "scheduler"], ),
                    "cols_num": ("INT", {"default": -1, "min": -1, "max": 1024 }),
                }}

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, images, params, order_by, cols_value, cols_num):
        from PIL import Image, ImageDraw, ImageFont
        import math

        if images.shape[0] != len(params):
            raise ValueError("Number of images and number of parameters do not match.")

        if order_by != "none":
            if cols_value != "none" and cols_num < 1:
                cols_num = len(set(p[cols_value] for p in params))
            sorted_params = sorted(params, key=lambda x: x[order_by])
            indices = [params.index(item) for item in sorted_params]
            params = sorted_params
            images = images[torch.tensor(indices)]

        width = images.shape[2]
        out_image = None

        font = ImageFont.truetype(os.path.join(FONTS_DIR, 'ShareTechMono-Regular.ttf'), min(48, int(32*(width/1024))))
        text_padding = 3
        line_height = font.getmask('WwMmQqlL1234567890').getbbox()[3] + font.getmetrics()[1] + text_padding*2

        for (image, param) in zip(images, params):
            text = f"time: {param['time']:.2f}s, seed: {param['seed']}, steps: {param['steps']}, denoise: {param['denoise']}\nsampler: {param['sampler']}, sched: {param['scheduler']}, sigmas at: {param['split_sigmas']}\nguidance: {param['guidance']}, max/base shift: {param['max_shift']}/{param['base_shift']}"
            lines = text.split("\n")
            text_height = line_height * len(lines)
            text_image = Image.new('RGB', (width, text_height), color=(0, 0, 0, 0))

            for i, line in enumerate(lines):
                draw = ImageDraw.Draw(text_image)
                draw.text((text_padding, i * line_height + text_padding), line, font=font, fill=(255, 255, 255))
            
            text_image = T.ToTensor()(text_image).unsqueeze(0).permute([0,2,3,1]).to(image.device)
            image = torch.cat([image.unsqueeze(0), text_image], 1)

            if out_image is None:
                out_image = image
            else:
                out_image = torch.cat([out_image, image], 0)

        if cols_num > -1:
            if cols_num == 0:
                mosaic_columns = int(math.sqrt(out_image.shape[0]))
                mosaic_columns = max(1, min(mosaic_columns, 1024))

            cols = min(mosaic_columns, out_image.shape[0])
            b, h, w, c = out_image.shape
            rows = math.ceil(b / cols)

            # Pad the tensor if necessary
            if b % cols != 0:
                padding = cols - (b % cols)
                out_image = F.pad(out_image, (0, 0, 0, 0, 0, 0, 0, padding))
                b = out_image.shape[0]
            
            # Reshape and transpose
            out_image = out_image.reshape(rows, cols, h, w, c)
            out_image = out_image.permute(0, 2, 1, 3, 4)
            out_image = out_image.reshape(rows * h, cols * w, c).unsqueeze(0)

            """
            width = out_image.shape[2]
            # add the title and notes on top
            if title and export_labels:
                title_font = ImageFont.truetype(os.path.join(FONTS_DIR, 'ShareTechMono-Regular.ttf'), 48)
                title_width = title_font.getbbox(title)[2]
                title_padding = 6
                title_line_height = title_font.getmask(title).getbbox()[3] + title_font.getmetrics()[1] + title_padding*2
                title_text_height = title_line_height
                title_text_image = Image.new('RGB', (width, title_text_height), color=(0, 0, 0, 0))

                draw = ImageDraw.Draw(title_text_image)
                draw.text((width//2 - title_width//2, title_padding), title, font=title_font, fill=(255, 255, 255))
                
                title_text_image = T.ToTensor()(title_text_image).unsqueeze(0).permute([0,2,3,1]).to(out_image.device)
                out_image = torch.cat([title_text_image, out_image], 1)
            """

        return (out_image, )

SAMPLING_CLASS_MAPPINGS = {
    "KSamplerVariationsStochastic+": KSamplerVariationsStochastic,
    "KSamplerVariationsWithNoise+": KSamplerVariationsWithNoise,
    "InjectLatentNoise+": InjectLatentNoise,
    "FluxSamplerParams+": FluxSamplerParams,
    "PlotParameters+": PlotParameters,
}

SAMPLING_NAME_MAPPINGS = {
    "KSamplerVariationsStochastic+": "🔧 KSampler Stochastic Variations",
    "KSamplerVariationsWithNoise+": "🔧 KSampler Variations with Noise Injection",
    "InjectLatentNoise+": "🔧 Inject Latent Noise",
    "FluxSamplerParams+": "🔧 Flux Sampler Parameters",
    "PlotParameters+": "🔧 Plot Sampler Parameters",
}