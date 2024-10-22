import os
import comfy.samplers
import comfy.sample
import torch
from nodes import common_ksampler, CLIPTextEncode
from comfy.utils import ProgressBar
from .utils import expand_mask, FONTS_DIR, parse_string_to_list
import torchvision.transforms.v2 as T
import torch.nn.functional as F
import logging
import folder_paths

# From https://github.com/BlenderNeko/ComfyUI_Noise/
def slerp(val, low, high):
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

        slerp_noise = slerp(variation_strength, base_noise, variation_noise)

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
                    "normalize": (["false", "true"], {"default": "false"}),
                },
                "optional": {
                    "mask": ("MASK", ),
                }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, latent, noise_seed, noise_strength, normalize="false", mask=None):
        torch.manual_seed(noise_seed)
        noise_latent = latent.copy()
        original_samples = noise_latent["samples"].clone()
        random_noise = torch.randn_like(original_samples)

        if normalize == "true":
            mean = original_samples.mean()
            std = original_samples.std()
            random_noise = random_noise * std + mean

        random_noise = original_samples + random_noise * noise_strength

        if mask is not None:
            mask = F.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(random_noise.shape[2], random_noise.shape[3]), mode="bilinear")
            mask = mask.expand((-1,random_noise.shape[1],-1,-1)).clamp(0.0, 1.0)
            if mask.shape[0] < random_noise.shape[0]:
                mask = mask.repeat((random_noise.shape[0] -1) // mask.shape[0] + 1, 1, 1, 1)[:random_noise.shape[0]]
            elif mask.shape[0] > random_noise.shape[0]:
                mask = mask[:random_noise.shape[0]]
            random_noise = mask * random_noise + (1-mask) * original_samples

        noise_latent["samples"] = random_noise

        return (noise_latent, )

class TextEncodeForSamplerParams:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Separate prompts with at least three dashes\n---\nLike so"}),
                "clip": ("CLIP", )
            }}

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, text, clip):
        import re
        output_text = []
        output_encoded = []
        text = re.sub(r'[-*=~]{4,}\n', '---\n', text)
        text = text.split("---\n")

        for t in text:
            t = t.strip()
            if t:
                output_text.append(t)
                output_encoded.append(CLIPTextEncode().encode(clip, t)[0])

        #if len(output_encoded) == 1:
        #    output = output_encoded[0]
        #else:
        output = {"text": output_text, "encoded": output_encoded}

        return (output, )

class SamplerSelectHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            **{s: ("BOOLEAN", { "default": False }) for s in comfy.samplers.KSampler.SAMPLERS},
        }}

    RETURN_TYPES = ("STRING", )
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, **values):
        values = [v for v in values if values[v]]
        values = ", ".join(values)

        return (values, )

class SchedulerSelectHelper:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            **{s: ("BOOLEAN", { "default": False }) for s in comfy.samplers.KSampler.SCHEDULERS},
        }}

    RETURN_TYPES = ("STRING", )
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, **values):
        values = [v for v in values if values[v]]
        values = ", ".join(values)

        return (values, )

class LorasForFluxParams:
    @classmethod
    def INPUT_TYPES(s):
        optional_loras = ['none'] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_1": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model_1": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "1.0" }),
            },
            #"optional": {
            #    "lora_2": (optional_loras, ),
            #    "strength_lora_2": ("STRING", { "multiline": False, "dynamicPrompts": False }),
            #    "lora_3": (optional_loras, ),
            #    "strength_lora_3": ("STRING", { "multiline": False, "dynamicPrompts": False }),
            #    "lora_4": (optional_loras, ),
            #    "strength_lora_4": ("STRING", { "multiline": False, "dynamicPrompts": False }),
            #}
        }

    RETURN_TYPES = ("LORA_PARAMS", )
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, lora_1, strength_model_1, lora_2="none", strength_lora_2="", lora_3="none", strength_lora_3="", lora_4="none", strength_lora_4=""):
        output = { "loras": [], "strengths": [] }
        output["loras"].append(lora_1)
        output["strengths"].append(parse_string_to_list(strength_model_1))

        if lora_2 != "none":
            output["loras"].append(lora_2)
            if strength_lora_2 == "":
                strength_lora_2 = "1.0"
            output["strengths"].append(parse_string_to_list(strength_lora_2))
        if lora_3 != "none":
            output["loras"].append(lora_3)
            if strength_lora_3 == "":
                strength_lora_3 = "1.0"
            output["strengths"].append(parse_string_to_list(strength_lora_3))
        if lora_4 != "none":
            output["loras"].append(lora_4)
            if strength_lora_4 == "":
                strength_lora_4 = "1.0"
            output["strengths"].append(parse_string_to_list(strength_lora_4))

        return (output,)


class FluxSamplerParams:
    def __init__(self):
        self.loraloader = None
        self.lora = (None, None)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "conditioning": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),

                    "seed": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "?" }),
                    "sampler": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "euler" }),
                    "scheduler": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "simple" }),
                    "steps": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "20" }),
                    "guidance": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "3.5" }),
                    "max_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
                    "base_shift": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "" }),
                    "denoise": ("STRING", { "multiline": False, "dynamicPrompts": False, "default": "1.0" }),
                },
                "optional": {
                    "loras": ("LORA_PARAMS",),
                }}

    RETURN_TYPES = ("LATENT","SAMPLER_PARAMS")
    RETURN_NAMES = ("latent", "params")
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, model, conditioning, latent_image, seed, sampler, scheduler, steps, guidance, max_shift, base_shift, denoise, loras=None):
        import random
        import time
        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise, BasicScheduler, BasicGuider, SamplerCustomAdvanced
        from comfy_extras.nodes_latent import LatentBatch
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux, ModelSamplingAuraFlow
        from node_helpers import conditioning_set_values
        from nodes import LoraLoader

        is_schnell = model.model.model_type == comfy.model_base.ModelType.FLOW

        noise = seed.replace("\n", ",").split(",")
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

        if steps == "":
            if is_schnell:
                steps = "4"
            else:
                steps = "20"
        steps = parse_string_to_list(steps)

        denoise = "1.0" if denoise == "" else denoise
        denoise = parse_string_to_list(denoise)

        guidance = "3.5" if guidance == "" else guidance
        guidance = parse_string_to_list(guidance)

        if not is_schnell:
            max_shift = "1.15" if max_shift == "" else max_shift
            base_shift = "0.5" if base_shift == "" else base_shift
        else:
            max_shift = "0"
            base_shift = "1.0" if base_shift == "" else base_shift

        max_shift = parse_string_to_list(max_shift)
        base_shift = parse_string_to_list(base_shift)

        cond_text = None
        if isinstance(conditioning, dict) and "encoded" in conditioning:
            cond_text = conditioning["text"]
            cond_encoded = conditioning["encoded"]
        else:
            cond_encoded = [conditioning]

        out_latent = None
        out_params = []

        basicschedueler = BasicScheduler()
        basicguider = BasicGuider()
        samplercustomadvanced = SamplerCustomAdvanced()
        latentbatch = LatentBatch()
        modelsamplingflux = ModelSamplingFlux() if not is_schnell else ModelSamplingAuraFlow()
        width = latent_image["samples"].shape[3]*8
        height = latent_image["samples"].shape[2]*8

        lora_strength_len = 1
        if loras:
            lora_model = loras["loras"]
            lora_strength = loras["strengths"]
            lora_strength_len = sum(len(i) for i in lora_strength)

            if self.loraloader is None:
                self.loraloader = LoraLoader()

        # count total number of samples
        total_samples = len(cond_encoded) * len(noise) * len(max_shift) * len(base_shift) * len(guidance) * len(sampler) * len(scheduler) * len(steps) * len(denoise) * lora_strength_len
        current_sample = 0
        if total_samples > 1:
            pbar = ProgressBar(total_samples)

        lora_strength_len = 1
        if loras:
            lora_strength_len = len(lora_strength[0])

        for los in range(lora_strength_len):
            if loras:
                patched_model = self.loraloader.load_lora(model, None, lora_model[0], lora_strength[0][los], 0)[0]
            else:
                patched_model = model

            for i in range(len(cond_encoded)):
                conditioning = cond_encoded[i]
                ct = cond_text[i] if cond_text else None
                for n in noise:
                    randnoise = Noise_RandomNoise(n)
                    for ms in max_shift:
                        for bs in base_shift:
                            if is_schnell:
                                work_model = modelsamplingflux.patch_aura(patched_model, bs)[0]
                            else:
                                work_model = modelsamplingflux.patch(patched_model, ms, bs, width, height)[0]
                            for g in guidance:
                                cond = conditioning_set_values(conditioning, {"guidance": g})
                                guider = basicguider.get_guider(work_model, cond)[0]
                                for s in sampler:
                                    samplerobj = comfy.samplers.sampler_object(s)
                                    for sc in scheduler:
                                        for st in steps:
                                            for d in denoise:
                                                sigmas = basicschedueler.get_sigmas(work_model, sc, st, d)[0]
                                                current_sample += 1
                                                log = f"Sampling {current_sample}/{total_samples} with seed {n}, sampler {s}, scheduler {sc}, steps {st}, guidance {g}, max_shift {ms}, base_shift {bs}, denoise {d}"
                                                lora_name = None
                                                lora_str = 0
                                                if loras:
                                                    lora_name = lora_model[0]
                                                    lora_str = lora_strength[0][los]
                                                    log += f", lora {lora_name}, lora_strength {lora_str}"
                                                logging.info(log)
                                                start_time = time.time()
                                                latent = samplercustomadvanced.sample(randnoise, guider, samplerobj, sigmas, latent_image)[1]
                                                elapsed_time = time.time() - start_time
                                                out_params.append({"time": elapsed_time,
                                                                "seed": n,
                                                                "width": width,
                                                                "height": height,
                                                                "sampler": s,
                                                                "scheduler": sc,
                                                                "steps": st,
                                                                "guidance": g,
                                                                "max_shift": ms,
                                                                "base_shift": bs,
                                                                "denoise": d,
                                                                "prompt": ct,
                                                                "lora": lora_name,
                                                                "lora_strength": lora_str})

                                                if out_latent is None:
                                                    out_latent = latent
                                                else:
                                                    out_latent = latentbatch.batch(out_latent, latent)[0]
                                                if total_samples > 1:
                                                    pbar.update(1)

        return (out_latent, out_params)

class PlotParameters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE", ),
                    "params": ("SAMPLER_PARAMS", ),
                    "order_by": (["none", "time", "seed", "steps", "denoise", "sampler", "scheduler", "guidance", "max_shift", "base_shift", "lora_strength"], ),
                    "cols_value": (["none", "time", "seed", "steps", "denoise", "sampler", "scheduler", "guidance", "max_shift", "base_shift", "lora_strength"], ),
                    "cols_num": ("INT", {"default": -1, "min": -1, "max": 1024 }),
                    "add_prompt": (["false", "true", "excerpt"], ),
                    "add_params": (["false", "true", "changes only"], {"default": "true"}),
                }}

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, images, params, order_by, cols_value, cols_num, add_prompt, add_params):
        from PIL import Image, ImageDraw, ImageFont
        import math
        import textwrap

        if images.shape[0] != len(params):
            raise ValueError("Number of images and number of parameters do not match.")

        _params = params.copy()

        if order_by != "none":
            sorted_params = sorted(_params, key=lambda x: x[order_by])
            indices = [_params.index(item) for item in sorted_params]
            images = images[torch.tensor(indices)]
            _params = sorted_params

        if cols_value != "none" and cols_num > -1:
            groups = {}
            for p in _params:
                value = p[cols_value]
                if value not in groups:
                    groups[value] = []
                groups[value].append(p)
            cols_num = len(groups)

            sorted_params = []
            groups = list(groups.values())
            for g in zip(*groups):
                sorted_params.extend(g)

            indices = [_params.index(item) for item in sorted_params]
            images = images[torch.tensor(indices)]
            _params = sorted_params
        elif cols_num == 0:
            cols_num = int(math.sqrt(images.shape[0]))
            cols_num = max(1, min(cols_num, 1024))

        width = images.shape[2]
        out_image = []

        font = ImageFont.truetype(os.path.join(FONTS_DIR, 'ShareTechMono-Regular.ttf'), min(48, int(32*(width/1024))))
        text_padding = 3
        line_height = font.getmask('Q').getbbox()[3] + font.getmetrics()[1] + text_padding*2
        char_width = font.getbbox('M')[2]+1 # using monospace font

        if add_params == "changes only":
            value_tracker = {}
            for p in _params:
                for key, value in p.items():
                    if key != "time":
                        if key not in value_tracker:
                            value_tracker[key] = set()
                        value_tracker[key].add(value)
            changing_keys = {key for key, values in value_tracker.items() if len(values) > 1 or key == "prompt"}

            result = []
            for p in _params:
                changing_params = {key: value for key, value in p.items() if key in changing_keys}
                result.append(changing_params)

            _params = result

        for (image, param) in zip(images, _params):
            image = image.permute(2, 0, 1)

            if add_params != "false":
                if add_params == "changes only":
                    text = "\n".join([f"{key}: {value}" for key, value in param.items() if key != "prompt"])
                else:
                    text = f"time: {param['time']:.2f}s, seed: {param['seed']}, steps: {param['steps']}, size: {param['width']}×{param['height']}\ndenoise: {param['denoise']}, sampler: {param['sampler']}, sched: {param['scheduler']}\nguidance: {param['guidance']}, max/base shift: {param['max_shift']}/{param['base_shift']}"
                    if 'lora' in param and param['lora']:
                        text += f"\nLoRA: {param['lora'][:32]}, str: {param['lora_strength']}"

                lines = text.split("\n")
                text_height = line_height * len(lines)
                text_image = Image.new('RGB', (width, text_height), color=(0, 0, 0))

                for i, line in enumerate(lines):
                    draw = ImageDraw.Draw(text_image)
                    draw.text((text_padding, i * line_height + text_padding), line, font=font, fill=(255, 255, 255))

                text_image = T.ToTensor()(text_image).to(image.device)
                image = torch.cat([image, text_image], 1)

            if 'prompt' in param and param['prompt'] and add_prompt != "false":
                prompt = param['prompt']
                if add_prompt == "excerpt":
                    prompt = " ".join(param['prompt'].split()[:64])
                    prompt += "..."

                cols = math.ceil(width / char_width)
                prompt_lines = textwrap.wrap(prompt, width=cols)
                prompt_height = line_height * len(prompt_lines)
                prompt_image = Image.new('RGB', (width, prompt_height), color=(0, 0, 0))

                for i, line in enumerate(prompt_lines):
                    draw = ImageDraw.Draw(prompt_image)
                    draw.text((text_padding, i * line_height + text_padding), line, font=font, fill=(255, 255, 255))

                prompt_image = T.ToTensor()(prompt_image).to(image.device)
                image = torch.cat([image, prompt_image], 1)

            # a little cleanup
            image = torch.nan_to_num(image, nan=0.0).clamp(0.0, 1.0)
            out_image.append(image)

        # ensure all images have the same height
        if add_prompt != "false" or add_params == "changes only":
            max_height = max([image.shape[1] for image in out_image])
            out_image = [F.pad(image, (0, 0, 0, max_height - image.shape[1])) for image in out_image]

        out_image = torch.stack(out_image, 0).permute(0, 2, 3, 1)

        # merge images
        if cols_num > -1:
            cols = min(cols_num, out_image.shape[0])
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

class GuidanceTimestepping:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "value": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, model, value, start_at, end_at):
        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        def apply_apg(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = args["cond_scale"]
            sigma = args["sigma"]

            sigma = sigma.detach().cpu()[0].item()

            if sigma <= sigma_start and sigma > sigma_end:
                cond_scale = value

            return uncond + (cond - uncond) * cond_scale
        
        m = model.clone()
        m.set_model_sampler_cfg_function(apply_apg)
        return (m,)

class ModelSamplingDiscreteFlowCustom(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        if model_config is not None:
            sampling_settings = model_config.sampling_settings
        else:
            sampling_settings = {}

        self.set_parameters(shift=sampling_settings.get("shift", 1.0), multiplier=sampling_settings.get("multiplier", 1000))

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000, cut_off=1.0, shift_multiplier=0):
        self.shift = shift
        self.multiplier = multiplier
        self.cut_off = cut_off
        self.shift_multiplier = shift_multiplier
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)
        self.register_buffer('sigmas', ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        shift = self.shift
        if timestep.dim() == 0:
            t = timestep.cpu().item() / self.multiplier
            if t <= self.cut_off:
                shift = shift * self.shift_multiplier
            
        return comfy.model_sampling.time_snr_shift(shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent

class ModelSamplingSD3Advanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.01}),
                              "cut_off": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step":0.05}),
                              "shift_multiplier": ("FLOAT", {"default": 2, "min": 0, "max": 10, "step":0.05}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"

    CATEGORY = "essentials/sampling"

    def execute(self, model, shift, multiplier=1000, cut_off=1.0, shift_multiplier=0):
        m = model.clone()
        

        sampling_base = ModelSamplingDiscreteFlowCustom
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier, cut_off=cut_off, shift_multiplier=shift_multiplier)
        m.add_object_patch("model_sampling", model_sampling)

        return (m, )

SAMPLING_CLASS_MAPPINGS = {
    "KSamplerVariationsStochastic+": KSamplerVariationsStochastic,
    "KSamplerVariationsWithNoise+": KSamplerVariationsWithNoise,
    "InjectLatentNoise+": InjectLatentNoise,
    "FluxSamplerParams+": FluxSamplerParams,
    "GuidanceTimestepping+": GuidanceTimestepping,
    "PlotParameters+": PlotParameters,
    "TextEncodeForSamplerParams+": TextEncodeForSamplerParams,
    "SamplerSelectHelper+": SamplerSelectHelper,
    "SchedulerSelectHelper+": SchedulerSelectHelper,
    "LorasForFluxParams+": LorasForFluxParams,
    "ModelSamplingSD3Advanced+": ModelSamplingSD3Advanced,
}

SAMPLING_NAME_MAPPINGS = {
    "KSamplerVariationsStochastic+": "🔧 KSampler Stochastic Variations",
    "KSamplerVariationsWithNoise+": "🔧 KSampler Variations with Noise Injection",
    "InjectLatentNoise+": "🔧 Inject Latent Noise",
    "FluxSamplerParams+": "🔧 Flux Sampler Parameters",
    "GuidanceTimestepping+": "🔧 Guidance Timestep (experimental)",
    "PlotParameters+": "🔧 Plot Sampler Parameters",
    "TextEncodeForSamplerParams+": "🔧Text Encode for Sampler Params",
    "SamplerSelectHelper+": "🔧 Sampler Select Helper",
    "SchedulerSelectHelper+": "🔧 Scheduler Select Helper",
    "LorasForFluxParams+": "🔧 LoRA for Flux Parameters",
    "ModelSamplingSD3Advanced+": "🔧 Model Sampling SD3 Advanced",
}