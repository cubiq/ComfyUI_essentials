import warnings
warnings.filterwarnings('ignore', module="torchvision")
import ast
import math
import random
import os
import operator as op
import numpy as np
import scipy
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter
import io

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

from nodes import MAX_RESOLUTION, SaveImage, common_ksampler
import folder_paths
import comfy.utils
import comfy.samplers
import comfy.sample

STOCHASTIC_SAMPLERS = ["euler_ancestral", "dpm_2_ancestral", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm"]

def p(image):
    return image.permute([0,3,1,2])
def pb(image):
    return image.permute([0,2,3,1])

# from https://github.com/pythongosssss/ComfyUI-Custom-Scripts
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any = AnyType("*")

EPSILON = 1e-5

class GetImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image):
        return (image.shape[2], image.shape[1],)

class ImageResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "keep_proportion": ("BOOLEAN", { "default": False }),
                "condition": (["always", "downscale if bigger", "upscale if smaller"],),
                "multiple_of": ("INT", { "default": 0, "min": 0, "max": 512, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, width, height, keep_proportion, interpolation="nearest", condition="always", multiple_of=0):
        _, oh, ow, _ = image.shape

        if keep_proportion is True:
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = ow

            #width = ow if width == 0 else width
            #height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow*ratio)
            height = round(oh*ratio)
        else:
            if width == 0:
                width = ow
            if height == 0:
                height = oh

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        outputs = p(image)

        if "always" in condition or ("bigger" in condition and (oh > height or ow > width)) or ("smaller" in condition and (oh < height or ow < width)):
            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

        outputs = pb(outputs)

        return(outputs, outputs.shape[2], outputs.shape[1],)

class ImageFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "axis": (["x", "y", "xy"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, axis):
        dim = ()
        if "y" in axis:
            dim += (1,)
        if "x" in axis:
            dim += (2,)
        image = torch.flip(image, dim)

        return(image,)

class ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"],),
                "x_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
                "y_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE","INT","INT",)
    RETURN_NAMES = ("IMAGE","x","y",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, width, height, position, x_offset, y_offset):
        _, oh, ow, _ = image.shape

        width = min(ow, width)
        height = min(oh, height)

        if "center" in position:
            x = round((ow-width) / 2)
            y = round((oh-height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh-height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow-width

        x += x_offset
        y += y_offset

        x2 = x+width
        y2 = y+height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        image = image[:, y:y2, x:x2, :]

        return(image, x, y, )

class ImageDesaturate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", { "default": 1.00, "min": 0.00, "max": 1.00, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, factor):
        grayscale = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        grayscale = (1.0 - factor) * image + factor * grayscale.unsqueeze(-1).repeat(1, 1, 1, 3)
        return(grayscale,)

class ImagePosterize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", { "default": 0.50, "min": 0.00, "max": 1.00, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, threshold):
        image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        #image = image.mean(dim=3, keepdim=True)
        image = (image > threshold).float()
        image = image.unsqueeze(-1).repeat(1, 1, 1, 3)

        return(image,)

class ImageEnhanceDifference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "exponent": ("FLOAT", { "default": 0.75, "min": 0.00, "max": 1.00, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image1, image2, exponent):
        if image1.shape != image2.shape:
            image2 = p(image2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = pb(image2)

        diff_image = image1 - image2
        diff_image = torch.pow(diff_image, exponent)
        diff_image = torch.clamp(diff_image, 0, 1)

        return(diff_image,)

class ImageExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, size, method):
        orig_size = image.shape[0]

        if orig_size == size:
            return (image,)

        if size <= 1:
            return (image[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(image.shape)[1:], dtype=image.dtype, device=image.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = image[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = image[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = image.repeat([math.ceil(size / image.shape[0])] + [1] * (len(image.shape) - 1))[:size]
        elif 'first' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat([image[:1].repeat(size-image.shape[0], 1, 1, 1), image], dim=0)
        elif 'last' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat((image, image[-1:].repeat((size-image.shape[0], 1, 1, 1))), dim=0)

        return (out,)

class ImageListToBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    INPUT_IS_LIST = True
    CATEGORY = "essentials"

    def execute(self, image):
        shape = image[0].shape[1:3]
        out = []

        for i in range(len(image)):
            img = p(image[i])
            if image[i].shape[1:3] != shape:
                transforms = T.Compose([
                    T.CenterCrop(min(img.shape[2], img.shape[3])),
                    T.Resize((shape[0], shape[1]), interpolation=T.InterpolationMode.BICUBIC),
                ])
                img = transforms(img)
            out.append(pb(img))
            #image[i] = pb(transforms(img))

        out = torch.cat(out, dim=0)

        return (out,)

class ExtractKeyframes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", { "default": 0.85, "min": 0.00, "max": 1.00, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("KEYFRAMES", "indexes")

    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, threshold):
        window_size = 2

        variations = torch.sum(torch.abs(image[1:] - image[:-1]), dim=[1, 2, 3])
        #variations = torch.sum((image[1:] - image[:-1]) ** 2, dim=[1, 2, 3])
        threshold = torch.quantile(variations.float(), threshold).item()

        keyframes = []
        for i in range(image.shape[0] - window_size + 1):
            window = image[i:i + window_size]
            variation = torch.sum(torch.abs(window[-1] - window[0])).item()

            if variation > threshold:
                keyframes.append(i + window_size - 1)

        return (image[keyframes], ','.join(map(str, keyframes)),)

class MaskFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "axis": (["x", "y", "xy"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, axis):
        dim = ()
        if "y" in axis:
            dim += (1,)
        if "x" in axis:
            dim += (2,)
        mask = torch.flip(mask, dims=dim)

        return(mask,)

class MaskBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("FLOAT", { "default": 6.0, "min": 0, "step": 0.5, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, amount):
        size = int(6 * amount +1)
        if size % 2 == 0:
            size+= 1
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        blurred = mask.unsqueeze(1)
        blurred = T.GaussianBlur(size, amount)(blurred)
        blurred = blurred.squeeze(1)

        return(blurred,)

class MaskPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",), },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)

class MaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask1, mask2):
        if mask1.shape[1:] != mask2.shape[1:]:
            mask2 = F.interpolate(mask2.unsqueeze(1), size=(mask1.shape[1], mask1.shape[2]), mode="bicubic").squeeze(1)

        out = torch.cat((mask1, mask2), dim=0)
        return (out,)

class MaskExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, size, method):
        orig_size = mask.shape[0]

        if orig_size == size:
            return (mask,)

        if size <= 1:
            return (mask[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(mask.shape)[1:], dtype=mask.dtype, device=mask.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = mask[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = mask[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = mask.repeat([math.ceil(size / mask.shape[0])] + [1] * (len(mask.shape) - 1))[:size]
        elif 'first' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat([mask[:1].repeat(size-mask.shape[0], 1, 1), mask], dim=0)
        elif 'last' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat((mask, mask[-1:].repeat((size-mask.shape[0], 1, 1))), dim=0)

        return (out,)

def cubic_bezier(t, p):
    p0, p1, p2, p3 = p
    return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

class MaskBoundingBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 128, "step": 1, }),
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("MASK", "IMAGE", "x", "y", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, padding, blur, image_optional=None):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image_optional is None:
            image_optional = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        # resize the image if it's not the same size as the mask
        if image_optional.shape[1] != mask.shape[1] or image_optional.shape[2] != mask.shape[2]:
            image_optional = p(image_optional)
            image_optional = comfy.utils.common_upscale(image_optional, mask.shape[2], mask.shape[1], upscale_method='bicubic', crop='center')
            image_optional = pb(image_optional)
        
        # match batch size
        if image_optional.shape[0] < mask.shape[0]:
            image_optional = torch.cat((image_optional, image_optional[-1].unsqueeze(0).repeat(mask.shape[0]-image_optional.shape[0], 1, 1, 1)), dim=0)
        elif image_optional.shape[0] > mask.shape[0]:
            image_optional = image_optional[:mask.shape[0]]

        # blur the mask
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask = T.functional.gaussian_blur(mask.unsqueeze(1), blur).squeeze(1)
        
        _, y, x = torch.where(mask)
        x1 = max(0, x.min().item() - padding)
        x2 = min(mask.shape[2], x.max().item() + 1 + padding)
        y1 = max(0, y.min().item() - padding)
        y2 = min(mask.shape[1], y.max().item() + 1 + padding)

        # crop the mask
        mask = mask[:, y1:y2, x1:x2]
        image_optional = image_optional[:, y1:y2, x1:x2, :]
        
        return (mask, image_optional, x1, y1, x2 - x1, y2 - y1)

class MaskFromColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "red": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "green": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "blue": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "threshold": ("INT", { "default": 0, "min": 0, "max": 127, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, red, green, blue, threshold):
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        color = torch.tensor([red, green, blue])
        lower_bound = (color - threshold).clamp(min=0)
        upper_bound = (color + threshold).clamp(max=255)
        lower_bound = lower_bound.view(1, 1, 1, 3)
        upper_bound = upper_bound.view(1, 1, 1, 3)
        mask = (temp >= lower_bound) & (temp <= upper_bound)
        mask = mask.all(dim=-1)
        mask = mask.float()

        return (mask, )

class MaskFromSegmentation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "segments": ("INT", { "default": 6, "min": 1, "max": 16, "step": 1, }),
                "remove_isolated_pixels": ("INT", { "default": 0, "min": 0, "max": 32, "step": 1, }),
                "remove_small_masks": ("FLOAT", { "default": 0.0, "min": 0., "max": 1., "step": 0.01, }),
                "fill_holes": ("BOOLEAN", { "default": False }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, segments, remove_isolated_pixels, fill_holes, remove_small_masks):
        im = image[0] # we only work on the first image in the batch
        im = Image.fromarray((im * 255).to(torch.uint8).cpu().numpy(), mode="RGB")
        im = im.quantize(palette=im.quantize(colors=segments), dither=Image.Dither.NONE)       
        im = torch.tensor(np.array(im.convert("RGB"))).float() / 255.0

        colors = im.reshape(-1, im.shape[-1])
        colors = torch.unique(colors, dim=0)

        masks = []
        for color in colors:
            mask = (im == color).all(dim=-1).float()
            # remove isolated pixels
            if remove_isolated_pixels > 0:
                mask_np = mask.cpu().numpy()
                mask_np = scipy.ndimage.binary_opening(mask_np, structure=np.ones((remove_isolated_pixels, remove_isolated_pixels)))
                mask = torch.from_numpy(mask_np)

            # fill holes
            if fill_holes:
                mask_np = mask.cpu().numpy()
                mask_np = scipy.ndimage.binary_fill_holes(mask_np)
                mask = torch.from_numpy(mask_np)

            # if the mask is too small, it's probably noise
            if mask.sum() / (mask.shape[0]*mask.shape[1]) > remove_small_masks:
                masks.append(mask)

        if masks == []:
            masks.append(torch.zeros_like(im).squeeze(-1).unsqueeze(0)) # return an empty mask if no masks were found, prevents errors

        mask = torch.stack(masks, dim=0).float()

        return (mask, )

class MaskFromRGBCMYBW:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "threshold_r": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_g": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_b": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "remove_isolated_pixels": ("INT", { "default": 0, "min": 0, "max": 32, "step": 1, }),
                "fill_holes": ("BOOLEAN", { "default": False }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, threshold_r, threshold_g, threshold_b, remove_isolated_pixels, fill_holes):
        red = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)).float()
        green = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
        blue = ((image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1-threshold_b)).float()

        cyan = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        magenta = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1-threshold_b)).float()
        yellow = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()

        black = ((image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)).float()
        white = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()

        if remove_isolated_pixels > 0 or fill_holes:
            colors = [red, green, blue, cyan, magenta, yellow, black, white]
            color_names = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white']
            processed_colors = {}

            for color_name, color in zip(color_names, colors):
                color = color.cpu().numpy()
                masks = []

                for i in range(image.shape[0]):
                    mask = color[i]
                    if remove_isolated_pixels > 0:
                        mask = scipy.ndimage.binary_opening(mask, structure=np.ones((remove_isolated_pixels, remove_isolated_pixels)))
                    if fill_holes:
                        mask = scipy.ndimage.binary_fill_holes(mask)
                    mask = torch.from_numpy(mask)
                    masks.append(mask)

                processed_colors[color_name] = torch.stack(masks, dim=0).float()

            red = processed_colors['red']
            green = processed_colors['green']
            blue = processed_colors['blue']
            cyan = processed_colors['cyan']
            magenta = processed_colors['magenta']
            yellow = processed_colors['yellow']
            black = processed_colors['black']
            white = processed_colors['white']

            del colors, processed_colors
        
        return (red, green, blue, cyan, magenta, yellow, black, white,)

class MaskSmooth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", { "default": 0, "min": 0, "max": 127, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, amount):
        if amount == 0:
            return (mask,)
        
        if amount % 2 == 0:
            amount += 1

        mask = mask > 0.5
        mask = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1).float()

        return (mask,)

class MaskFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", ),
                "start": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "length": ("INT", { "default": -1, "min": -1, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, start, length):
        if length<0:
            length = mask.shape[0]
        start = min(start, mask.shape[0]-1)
        length = min(mask.shape[0]-start, length)
        return (mask[start:start + length], )

class MaskFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "values": ("FLOAT", { "min": 0.0, "max": 1.0, "step": 0.01, }),
                "width": ("INT", { "default": 32, "min": 1, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 32, "min": 1, "max": MAX_RESOLUTION, "step": 8, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, values, width, height):
        if not isinstance(values, list):
            values = [values]

        values = torch.tensor(values).float()
        values = torch.clamp(values, 0.0, 1.0)
        #values = (values - values.min()) / values.max()

        return (values.unsqueeze(1).unsqueeze(2).repeat(1, width, height), )

class ImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "start": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "length": ("INT", { "default": -1, "min": -1, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, start, length):
        if length<0:
            length = image.shape[0]
        start = min(start, image.shape[0]-1)
        length = min(image.shape[0]-start, length)
        return (image[start:start + length], )

class ImageCompositeFromMaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_from": ("IMAGE", ),
                "image_to": ("IMAGE", ),
                "mask": ("MASK", )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image_from, image_to, mask):
        frames = mask.shape[0]

        if image_from.shape[1] != image_to.shape[1] or image_from.shape[2] != image_to.shape[2]:
            image_to = p(image_to)
            image_to = comfy.utils.common_upscale(image_to, image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center')
            image_to = pb(image_to)

        if frames < image_from.shape[0]:
            image_from = image_from[:frames]
        elif frames > image_from.shape[0]:
            image_from = torch.cat((image_from, image_from[-1].unsqueeze(0).repeat(frames-image_from.shape[0], 1, 1, 1)), dim=0)

        mask = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        if image_from.shape[1] != mask.shape[1] or image_from.shape[2] != mask.shape[2]:
            mask = p(mask)
            mask = comfy.utils.common_upscale(mask, image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center')
            mask = pb(mask)

        out = mask * image_to + (1 - mask) * image_from

        return (out, )

class TransitionMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "frames": ("INT", { "default": 16, "min": 1, "max": 9999, "step": 1, }),
                "start_frame": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "end_frame": ("INT", { "default": 9999, "min": 0, "step": 1, }),
                "transition_type": (["horizontal slide", "vertical slide", "horizontal bar", "vertical bar", "center box", "horizontal door", "vertical door", "circle", "fade"],),
                "timing_function": (["linear", "in", "out", "in-out"],)
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, width, height, frames, start_frame, end_frame, transition_type, timing_function):
        if timing_function == 'in':
            tf = [0.0, 0.0, 0.5, 1.0]
        elif timing_function == 'out':
            tf = [0.0, 0.5, 1.0, 1.0]
        elif timing_function == 'in-out':
            tf = [0, 1, 0, 1]
        #elif timing_function == 'back':
        #    tf = [0, 1.334, 1.334, 0]
        else:
            tf = [0, 0, 1, 1]

        out = []

        end_frame = min(frames, end_frame)
        transition = end_frame - start_frame

        if start_frame > 0:
            out = out + [torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")] * start_frame

        for i in range(transition):
            frame = torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")
            progress = i/(transition-1)

            if timing_function != 'linear':
                progress = cubic_bezier(progress, tf)

            if "horizontal slide" in transition_type:
                pos = round(width*progress)
                frame[:, :pos] = 1.0
            elif "vertical slide" in transition_type:
                pos = round(height*progress)
                frame[:pos, :] = 1.0
            elif "box" in transition_type:
                box_w = round(width*progress)
                box_h = round(height*progress)
                x1 = (width - box_w) // 2
                y1 = (height - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                frame[y1:y2, x1:x2] = 1.0
            elif "circle" in transition_type:
                radius = math.ceil(math.sqrt(pow(width,2)+pow(height,2))*progress/2)
                c_x = width // 2
                c_y = height // 2
                # is this real life? Am I hallucinating?
                x = torch.arange(0, width, dtype=torch.float32, device="cpu")
                y = torch.arange(0, height, dtype=torch.float32, device="cpu")
                y, x = torch.meshgrid((y, x), indexing="ij")
                circle = ((x - c_x) ** 2 + (y - c_y) ** 2) <= (radius ** 2)
                frame[circle] = 1.0
            elif "horizontal bar" in transition_type:
                bar = round(height*progress)
                y1 = (height - bar) // 2
                y2 = y1 + bar
                frame[y1:y2, :] = 1.0
            elif "vertical bar" in transition_type:
                bar = round(width*progress)
                x1 = (width - bar) // 2
                x2 = x1 + bar
                frame[:, x1:x2] = 1.0
            elif "horizontal door" in transition_type:
                bar = math.ceil(height*progress/2)
                if bar > 0:
                    frame[:bar, :] = 1.0
                    frame[-bar:, :] = 1.0
            elif "vertical door" in transition_type:
                bar = math.ceil(width*progress/2)
                if bar > 0:
                    frame[:, :bar] = 1.0
                    frame[:, -bar:] = 1.0
            elif "fade" in transition_type:
                frame[:,:] = progress

            out.append(frame)

        if end_frame < frames:
            out = out + [torch.full((height, width), 1.0, dtype=torch.float32, device="cpu")] * (frames - end_frame)

        out = torch.stack(out, dim=0)

        return (out, )

def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)

def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)

# From https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/
class ImageCAS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials"
    FUNCTION = "execute"

    def execute(self, image, amount):
        img = F.pad(p(image), pad=(1, 1, 1, 1)).cpu()

        a = img[..., :-2, :-2]
        b = img[..., :-2, 1:-1]
        c = img[..., :-2, 2:]
        d = img[..., 1:-1, :-2]
        e = img[..., 1:-1, 1:-1]
        f = img[..., 1:-1, 2:]
        g = img[..., 2:, :-2]
        h = img[..., 2:, 1:-1]
        i = img[..., 2:, 2:]

        # Computing contrast
        cross = (b, d, e, f, h)
        mn = min_(cross)
        mx = max_(cross)

        diag = (a, c, g, i)
        mn2 = min_(diag)
        mx2 = max_(diag)
        mx = mx + mx2
        mn = mn + mn2

        # Computing local weight
        inv_mx = torch.reciprocal(mx + EPSILON)
        amp = inv_mx * torch.minimum(mn, (2 - mx))

        # scaling
        amp = torch.sqrt(amp)
        w = - amp * (amount * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4*w)

        output = ((b + d + f + h)*w + e) * div
        output = output.clamp(0, 1)
        #output = torch.nan_to_num(output)   # this seems the only way to ensure there are no NaNs

        output = pb(output)

        return (output,)

operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}

op_functions = {
    'min': min,
    'max': max
}

class SimpleMath:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "a": ("INT,FLOAT", { "default": 0.0, "step": 0.1 }),
                "b": ("INT,FLOAT", { "default": 0.0, "step": 0.1 }),
            },
            "required": {
                "value": ("STRING", { "multiline": False, "default": "" }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", )
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, value, a = 0.0, b = 0.0):
        def eval_(node):
            if isinstance(node, ast.Num): # number
                return node.n
            elif isinstance(node, ast.Name): # variable
                if node.id == "a":
                    return a
                if node.id == "b":
                    return b
            elif isinstance(node, ast.BinOp): # <left> <operator> <right>
                return operators[type(node.op)](eval_(node.left), eval_(node.right))
            elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
                return operators[type(node.op)](eval_(node.operand))
            elif isinstance(node, ast.Call): # custom function
                if node.func.id in op_functions:
                    args =[eval_(arg) for arg in node.args]
                    return op_functions[node.func.id](*args)
            else:
                return 0

        result = eval_(ast.parse(value, mode='eval').body)

        if math.isnan(result):
            result = 0.0

        return (round(result), result, )

class ModelCompile():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "fullgraph": ("BOOLEAN", { "default": False }),
                "dynamic": ("BOOLEAN", { "default": False }),
                "mode": (["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],),
            },
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, model, fullgraph, dynamic, mode):
        work_model = model.clone()
        torch._dynamo.config.suppress_errors = True
        work_model.model.diffusion_model = torch.compile(work_model.model.diffusion_model, dynamic=dynamic, fullgraph=fullgraph, mode=mode)
        return( work_model, )

class ConsoleDebug:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": (any, {}),
            },
            "optional": {
                "prefix": ("STRING", { "multiline": False, "default": "Value:" })
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "essentials"
    OUTPUT_NODE = True

    def execute(self, value, prefix):
        print(f"\033[96m{prefix} {value}\033[0m")

        return (None,)

class DebugTensorShape:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (any, {}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "essentials"
    OUTPUT_NODE = True

    def execute(self, tensor):
        shapes = []
        def tensorShape(tensor):
            if isinstance(tensor, dict):
                for k in tensor:
                    tensorShape(tensor[k])
            elif isinstance(tensor, list):
                for i in range(len(tensor)):
                    tensorShape(tensor[i])
            elif hasattr(tensor, 'shape'):
                shapes.append(list(tensor.shape))

        tensorShape(tensor)

        print(f"\033[96mShapes found: {shapes}\033[0m")

        return (None,)

class BatchCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch": (any, {}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, batch):
        count = 0
        if hasattr(batch, 'shape'):
            count = batch.shape[0]
        elif isinstance(batch, dict) and 'samples' in batch:
            count = batch['samples'].shape[0]
        elif isinstance(batch, list) or isinstance(batch, dict):
            count = len(batch)

        return (count, )

class ImageSeamCarving:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "energy": (["backward", "forward"],),
                "order": (["width-first", "height-first"],),
            },
            "optional": {
                "keep_mask": ("MASK",),
                "drop_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials"
    FUNCTION = "execute"

    def execute(self, image, width, height, energy, order, keep_mask=None, drop_mask=None):
        try:
            from .carve import seam_carving
        except ImportError as e:
            raise Exception(e)

        img = p(image)

        if keep_mask is not None:
            #keep_mask = keep_mask.reshape((-1, 1, keep_mask.shape[-2], keep_mask.shape[-1])).movedim(1, -1)
            keep_mask = p(keep_mask.unsqueeze(-1))

            if keep_mask.shape[2] != img.shape[2] or keep_mask.shape[3] != img.shape[3]:
                keep_mask = F.interpolate(keep_mask, size=(img.shape[2], img.shape[3]), mode="bilinear")
        if drop_mask is not None:
            drop_mask = p(drop_mask.unsqueeze(-1))

            if drop_mask.shape[2] != img.shape[2] or drop_mask.shape[3] != img.shape[3]:
                drop_mask = F.interpolate(drop_mask, size=(img.shape[2], img.shape[3]), mode="bilinear")

        out = []
        for i in range(img.shape[0]):
            resized = seam_carving(
                T.ToPILImage()(img[i]),
                size=(width, height),
                energy_mode=energy,
                order=order,
                keep_mask=T.ToPILImage()(keep_mask[i]) if keep_mask is not None else None,
                drop_mask=T.ToPILImage()(drop_mask[i]) if drop_mask is not None else None,
            )
            out.append(T.ToTensor()(resized))

        out = torch.stack(out)
        out = pb(out)

        return(out, )

class CLIPTextEncodeSDXLSimplified:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "size_cond_factor": ("INT", {"default": 4, "min": 1, "max": 16 }),
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            "clip": ("CLIP", ),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, clip, width, height, size_cond_factor, text):
        crop_w = 0
        crop_h = 0
        width = width*size_cond_factor
        height = height*size_cond_factor
        target_width = width
        target_height = height
        text_g = text_l = text

        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]], )

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
    CATEGORY = "essentials"

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

def prepare_mask(mask, shape):
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
    mask = mask.expand((-1,shape[1],-1,-1))
    if mask.shape[0] < shape[0]:
        mask = mask.repeat((shape[0] -1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
    return mask

def expand_mask(mask, expand, tapered_corners):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)

    return torch.stack(out, dim=0)

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
    CATEGORY = "essentials"

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
            noise_mask = prepare_mask(latent_image["noise_mask"], latent_image['samples'].shape)
            work_latent["samples"] = noise_mask * work_latent["samples"] + (1-noise_mask) * latent_image["samples"]
            work_latent['noise_mask'] = expand_mask(latent_image["noise_mask"].clone(), 5, True)

        return common_ksampler(model, main_seed, steps, cfg, sampler_name, scheduler, positive, negative, work_latent, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

class SDXLEmptyLatentSizePicker:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "resolution": (["704x1408 (0.5)","704x1344 (0.52)","768x1344 (0.57)","768x1280 (0.6)","832x1216 (0.68)","832x1152 (0.72)","896x1152 (0.78)","896x1088 (0.82)","960x1088 (0.88)","960x1024 (0.94)","1024x1024 (1.0)","1024x960 (1.07)","1088x960 (1.13)","1088x896 (1.21)","1152x896 (1.29)","1152x832 (1.38)","1216x832 (1.46)","1280x768 (1.67)","1344x768 (1.75)","1344x704 (1.91)","1408x704 (2.0)","1472x704 (2.09)","1536x640 (2.4)","1600x640 (2.5)","1664x576 (2.89)","1728x576 (3.0)",], {"default": "1024x1024 (1.0)"}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }}

    RETURN_TYPES = ("LATENT","INT","INT",)
    RETURN_NAMES = ("LATENT","width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, resolution, batch_size):
        width, height = resolution.split(" ")[0].split("x")
        width = int(width)
        height = int(height)

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return ({"samples":latent}, width, height,)

LUTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "luts")
# From https://github.com/yoonsikp/pycubelut/blob/master/pycubelut.py (MIT license)
class ImageApplyLUT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": ([f for f in os.listdir(LUTS_DIR) if f.endswith('.cube')], ),
                "log_colorspace": ("BOOLEAN", { "default": False }),
                "clip_values": ("BOOLEAN", { "default": False }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1 }),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    # TODO: check if we can do without numpy
    def execute(self, image, lut_file, log_colorspace, clip_values, strength):
        from colour.io.luts.iridas_cube import read_LUT_IridasCube

        lut = read_LUT_IridasCube(os.path.join(LUTS_DIR, lut_file))
        lut.name = lut_file

        if clip_values:
            if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
                lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
            else:
                if len(lut.table.shape) == 2:  # 3x1D
                    for dim in range(3):
                        lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
                else:  # 3D
                    for dim in range(3):
                        lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

        out = []
        for img in image: # TODO: is this more resource efficient? should we use a batch instead?
            lut_img = img.numpy().copy()

            is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
            dom_scale = None
            if is_non_default_domain:
                dom_scale = lut.domain[1] - lut.domain[0]
                lut_img = lut_img * dom_scale + lut.domain[0]
            if log_colorspace:
                lut_img = lut_img ** (1/2.2)
            lut_img = lut.apply(lut_img)
            if log_colorspace:
                lut_img = lut_img ** (2.2)
            if is_non_default_domain:
                lut_img = (lut_img - lut.domain[0]) / dom_scale

            lut_img = torch.from_numpy(lut_img)
            if strength < 1.0:
                lut_img = strength * lut_img + (1 - strength) * img
            out.append(lut_img)

        out = torch.stack(out)

        return (out, )

FONTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "fonts")
class DrawText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", { "multiline": True, "dynamicPrompts": True, "default": "Hello, World!" }),
                "font": ([f for f in os.listdir(FONTS_DIR) if f.endswith('.ttf') or f.endswith('.otf')], ),
                "size": ("INT", { "default": 56, "min": 1, "max": 9999, "step": 1 }),
                "color": ("STRING", { "multiline": False, "default": "#FFFFFF" }),
                "background_color": ("STRING", { "multiline": False, "default": "#00000000" }),
                "shadow_distance": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1 }),
                "shadow_blur": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1 }),
                "shadow_color": ("STRING", { "multiline": False, "default": "#000000" }),
                "alignment": (["left", "center", "right"],),
                "width": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1 }),
                "height": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1 }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, text, font, size, color, background_color, shadow_distance, shadow_blur, shadow_color, alignment, width, height):
        font = ImageFont.truetype(os.path.join(FONTS_DIR, font), size)

        lines = text.split("\n")

        # Calculate the width and height of the text
        text_width = max(font.getbbox(line)[2] for line in lines)
        line_height = font.getmask(text).getbbox()[3] + font.getmetrics()[1]  # add descent to height
        text_height = line_height * len(lines)

        width = width if width > 0 else text_width
        height = height if height > 0 else text_height

        background_color = ImageColor.getrgb(background_color)
        image = Image.new('RGBA', (width + shadow_distance, height + shadow_distance), color=background_color)

        image_shadow = None
        if shadow_distance > 0:
            image_shadow = Image.new('RGBA', (width + shadow_distance, height + shadow_distance), color=background_color)

        for i, line in enumerate(lines):
            line_width = font.getbbox(line)[2]
            #text_height =font.getbbox(line)[3]
            if alignment == "left":
                x = 0
            elif alignment == "center":
                x = (width - line_width) / 2
            elif alignment == "right":
                x = width - line_width
            y = i * line_height

            draw = ImageDraw.Draw(image)
            draw.text((x, y), line, font=font, fill=color)

            if image_shadow is not None:
                draw = ImageDraw.Draw(image_shadow)
                draw.text((x + shadow_distance, y + shadow_distance), line, font=font, fill=shadow_color)

        if image_shadow is not None:
            image_shadow = image_shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
            image = Image.alpha_composite(image_shadow, image)

        image = pb(T.ToTensor()(image).unsqueeze(0))
        mask = image[:, :, :, 3] if image.shape[3] == 4 else torch.ones_like(image[:, :, :, 0])

        return (image[:, :, :, :3], mask,)

class RemBGSession:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["u2net: general purpose", "u2netp: lightweight general purpose", "u2net_human_seg: human segmentation", "u2net_cloth_seg: cloths Parsing", "silueta: very small u2net", "isnet-general-use: general purpose", "isnet-anime: anime illustrations", "sam: general purpose"],),
                "providers": (['CPU', 'CUDA', 'ROCM', 'DirectML', 'OpenVINO', 'CoreML', 'Tensorrt', 'Azure'],),
            },
        }

    RETURN_TYPES = ("REMBG_SESSION",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, model, providers):
        from rembg import new_session as rembg_new_session

        model = model.split(":")[0]
        return (rembg_new_session(model, providers=[providers+"ExecutionProvider"]),)

class ImageRemoveBackground:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rembg_session": ("REMBG_SESSION",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, rembg_session, image):
        from rembg import remove as rembg

        image = p(image)
        output = []
        for img in image:
            img = T.ToPILImage()(img)
            img = rembg(img, session=rembg_session)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0)
        output = pb(output)
        mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])

        return(output[:, :, :, :3], mask,)

class PixelOEPixelize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_mode": (["contrast", "bicubic", "nearest", "center", "k-centroid"],),
                "target_size": ("INT", { "default": 128, "min": 1, "max": MAX_RESOLUTION, "step": 16 }),
                "patch_size": ("INT", { "default": 16, "min": 4, "max": 32, "step": 2 }),
                "thickness": ("INT", { "default": 2, "min": 1, "max": 16, "step": 1 }),
                #"contrast": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                #"saturation": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "color_matching": ("BOOLEAN", { "default": True }),
                "upscale": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, downscale_mode, target_size, patch_size, thickness, color_matching, upscale):
        from pixeloe.pixelize import pixelize

        image = image.clone().mul(255).clamp(0, 255).byte().cpu().numpy()
        output = []
        for img in image:
            img = pixelize(img,
                           mode=downscale_mode,
                           target_size=target_size,
                           patch_size=patch_size,
                           thickness=thickness,
                           contrast=1.0,
                           saturation=1.0,
                           color_matching=color_matching,
                           no_upscale=not upscale)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0)
        output = pb(output)

        return(output,)

class NoiseFromImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_size": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "color_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_scale_diff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "noise_strenght": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "saturation": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1 }),
            },
            "optional": {
                "noise_mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, noise_size, color_noise, mask_strength, mask_scale_diff, noise_strenght, saturation, contrast, blur, noise_mask=None):
        torch.manual_seed(0)

        elastic_alpha = max(image.shape[1], image.shape[2])# * noise_size
        elastic_sigma = elastic_alpha / 400 * noise_size

        blur_size = int(6 * blur+1)
        if blur_size % 2 == 0:
            blur_size+= 1

        if noise_mask is None:
            noise_mask = image

        # Ensure noise mask is the same size as the image
        if noise_mask.shape[1:] != image.shape[1:]:
            noise_mask = F.interpolate(p(noise_mask), size=(image.shape[1], image.shape[2]), mode='bicubic', align_corners=False)
            noise_mask = pb(noise_mask)
        # Ensure we have the same number of masks and images
        if noise_mask.shape[0] > image.shape[0]:
            noise_mask = noise_mask[:image.shape[0]]
        else:
            noise_mask = torch.cat((noise_mask, noise_mask[-1:].repeat((image.shape[0]-noise_mask.shape[0], 1, 1, 1))), dim=0)

        # Convert image to grayscale mask
        noise_mask = noise_mask.mean(dim=3).unsqueeze(-1)

        # add color noise
        imgs = p(image.clone())
        if color_noise > 0:
            color_noise = torch.normal(torch.zeros_like(imgs), std=color_noise)

            #color_noise = torch.rand_like(imgs) * (color_noise * 2) - color_noise

            color_noise *= (imgs - imgs.min()) / (imgs.max() - imgs.min())

            imgs = imgs + color_noise
            imgs = imgs.clamp(0, 1)

        # create fine noise
        fine_noise = []
        for n in imgs:
            avg_color = n.mean(dim=[1,2])

            tmp_noise = T.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, fill=avg_color.tolist())(n)
            #tmp_noise = T.functional.adjust_saturation(tmp_noise, 2.0)
            tmp_noise = T.GaussianBlur(blur_size, blur)(tmp_noise)
            tmp_noise = T.ColorJitter(contrast=(contrast,contrast), saturation=(saturation,saturation))(tmp_noise)
            fine_noise.append(tmp_noise)

            #tmp_noise = F.interpolate(tmp_noise, scale_factor=.1, mode='bilinear', align_corners=False)
            #tmp_noise = F.interpolate(tmp_noise, size=(tmp_noise.shape[1], tmp_noise.shape[2]), mode='bilinear', align_corners=False)

            #tmp_noise = T.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma/3, fill=avg_color.tolist())(n)
            #tmp_noise = T.GaussianBlur(blur_size, blur)(tmp_noise)
            #tmp_noise = T.functional.adjust_saturation(tmp_noise, saturation)
            #tmp_noise = T.ColorJitter(contrast=(contrast,contrast), saturation=(saturation,saturation))(tmp_noise)
            #fine_noise.append(tmp_noise)

        imgs = None
        del imgs

        fine_noise = torch.stack(fine_noise, dim=0)
        fine_noise = pb(fine_noise)
        #fine_noise = torch.stack(fine_noise, dim=0)
        #fine_noise = pb(fine_noise)
        mask_scale_diff = min(mask_scale_diff, 0.99)
        if mask_scale_diff > 0:
            coarse_noise = F.interpolate(p(fine_noise), scale_factor=1-mask_scale_diff, mode='area')
            coarse_noise = F.interpolate(coarse_noise, size=(fine_noise.shape[1], fine_noise.shape[2]), mode='bilinear', align_corners=False)
            coarse_noise = pb(coarse_noise)
        else:
            coarse_noise = fine_noise

        #noise_mask = noise_mask * mask_strength + (1 - mask_strength)
        # merge fine and coarse noise
        output = (1 - noise_mask) * coarse_noise + noise_mask * fine_noise
        #noise_mask = noise_mask * mask_strength
        if mask_strength < 1:
            noise_mask = noise_mask.pow(mask_strength)
            noise_mask = torch.nan_to_num(noise_mask).clamp(0, 1)
            output = noise_mask * output + (1 - noise_mask) * image

        # apply noise to image
        output = output * noise_strenght + image * (1 - noise_strenght)
        output = output.clamp(0, 1)

        return (output,noise_mask.repeat(1,1,1,3),)

class RemoveLatentMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    CATEGORY = "essentials"

    def execute(self, samples):
        s = samples.copy()
        if "noise_mask" in s:
            del s["noise_mask"]

        return (s,)

class ConditioningCombineMultiple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
            }, "optional": {
                "conditioning_3": ("CONDITIONING",),
                "conditioning_4": ("CONDITIONING",),
                "conditioning_5": ("CONDITIONING",),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, conditioning_1, conditioning_2, conditioning_3=None, conditioning_4=None, conditioning_5=None):
        c = conditioning_1 + conditioning_2

        if conditioning_3 is not None:
            c += conditioning_3
        if conditioning_4 is not None:
            c += conditioning_4
        if conditioning_5 is not None:
            c += conditioning_5
        
        return (c,)

class ImageBatchMultiple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], { "default": "lanczos" }),
            }, "optional": {
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image_1, image_2, method, image_3=None, image_4=None, image_5=None):
        if image_1.shape[1:] != image_2.shape[1:]:
            image_2 = comfy.utils.common_upscale(image_2.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
        out = torch.cat((image_1, image_2), dim=0)

        if image_3 is not None:
            if image_1.shape[1:] != image_3.shape[1:]:
                image_3 = comfy.utils.common_upscale(image_3.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_3), dim=0)
        if image_4 is not None:
            if image_1.shape[1:] != image_4.shape[1:]:
                image_4 = comfy.utils.common_upscale(image_4.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_4), dim=0)
        if image_5 is not None:
            if image_1.shape[1:] != image_5.shape[1:]:
                image_5 = comfy.utils.common_upscale(image_5.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_5), dim=0)
        
        return (out,)


NODE_CLASS_MAPPINGS = {
    "GetImageSize+": GetImageSize,

    "ImageResize+": ImageResize,
    "ImageCrop+": ImageCrop,
    "ImageFlip+": ImageFlip,

    "ImageDesaturate+": ImageDesaturate,
    "ImagePosterize+": ImagePosterize,
    "ImageCASharpening+": ImageCAS,
    "ImageSeamCarving+": ImageSeamCarving,
    "ImageEnhanceDifference+": ImageEnhanceDifference,
    "ImageExpandBatch+": ImageExpandBatch,
    "ImageFromBatch+": ImageFromBatch,
    "ImageListToBatch+": ImageListToBatch,
    "ImageCompositeFromMaskBatch+": ImageCompositeFromMaskBatch,
    "ExtractKeyframes+": ExtractKeyframes,
    "ImageApplyLUT+": ImageApplyLUT,
    "PixelOEPixelize+": PixelOEPixelize,

    "MaskBlur+": MaskBlur,
    "MaskFlip+": MaskFlip,
    "MaskPreview+": MaskPreview,
    "MaskBatch+": MaskBatch,
    "MaskExpandBatch+": MaskExpandBatch,
    "TransitionMask+": TransitionMask,
    "MaskFromColor+": MaskFromColor,
    "MaskFromBatch+": MaskFromBatch,
    "MaskBoundingBox+": MaskBoundingBox,
    "MaskFromSegmentation+": MaskFromSegmentation,
    "MaskFromRGBCMYBW+": MaskFromRGBCMYBW,
    "MaskSmooth+": MaskSmooth,
    "MaskFromList+": MaskFromList,

    "SimpleMath+": SimpleMath,
    "ConsoleDebug+": ConsoleDebug,
    "DebugTensorShape+": DebugTensorShape,

    "ModelCompile+": ModelCompile,
    "BatchCount+": BatchCount,

    "KSamplerVariationsStochastic+": KSamplerVariationsStochastic,
    "KSamplerVariationsWithNoise+": KSamplerVariationsWithNoise,
    "CLIPTextEncodeSDXL+": CLIPTextEncodeSDXLSimplified,
    "SDXLEmptyLatentSizePicker+": SDXLEmptyLatentSizePicker,

    "DrawText+": DrawText,
    "RemBGSession+": RemBGSession,
    "ImageRemoveBackground+": ImageRemoveBackground,

    "RemoveLatentMask+": RemoveLatentMask,
    "ConditioningCombineMultiple+": ConditioningCombineMultiple,
    "ImageBatchMultiple+": ImageBatchMultiple,

    #"NoiseFromImage~": NoiseFromImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetImageSize+": "🔧 Get Image Size",
    "ImageResize+": "🔧 Image Resize",
    "ImageCrop+": "🔧 Image Crop",
    "ImageFlip+": "🔧 Image Flip",

    "ImageDesaturate+": "🔧 Image Desaturate",
    "ImagePosterize+": "🔧 Image Posterize",
    "ImageCASharpening+": "🔧 Image Contrast Adaptive Sharpening",
    "ImageSeamCarving+": "🔧 Image Seam Carving",
    "ImageEnhanceDifference+": "🔧 Image Enhance Difference",
    "ImageExpandBatch+": "🔧 Image Expand Batch",
    "ImageFromBatch+": "🔧 Image From Batch",
    "ImageListToBatch+": "🔧 Image List To Batch",
    "ImageCompositeFromMaskBatch+": "🔧 Image Composite From Mask Batch",
    "ExtractKeyframes+": "🔧 Extract Keyframes (experimental)",
    "ImageApplyLUT+": "🔧 Image Apply LUT",
    "PixelOEPixelize+": "🔧 Pixelize",

    "MaskBlur+": "🔧 Mask Blur",
    "MaskFlip+": "🔧 Mask Flip",
    "MaskPreview+": "🔧 Mask Preview",
    "MaskBatch+": "🔧 Mask Batch",
    "MaskExpandBatch+": "🔧 Mask Expand Batch",
    "TransitionMask+": "🔧 Transition Mask",
    "MaskFromColor+": "🔧 Mask From Color",
    "MaskFromBatch+": "🔧 Mask From Batch",
    "MaskBoundingBox+": "🔧 Mask Bounding Box",
    "MaskFromSegmentation+": "🔧 Mask From Segmentation",
    "MaskFromRGBCMYBW+": "🔧 Mask From RGB/CMY/BW",
    "MaskSmooth+": "🔧 Mask Smooth",
    "MaskFromList+": "🔧 Mask From List",

    "SimpleMath+": "🔧 Simple Math",
    "ConsoleDebug+": "🔧 Console Debug",
    "DebugTensorShape+": "🔧 Tensor Shape Debug",

    "ModelCompile+": "🔧 Compile Model",
    "BatchCount+": "🔧 Batch Count",

    "KSamplerVariationsStochastic+": "🔧 KSampler Stochastic Variations",
    "KSamplerVariationsWithNoise+": "🔧 KSampler Variations with Noise Injection",
    "CLIPTextEncodeSDXL+": "🔧 SDXLCLIPTextEncode",
    "SDXLEmptyLatentSizePicker+": "🔧 SDXL Empty Latent Size Picker",

    "DrawText+": "🔧 Draw Text",
    "RemBGSession+": "🔧 RemBG Session",
    "ImageRemoveBackground+": "🔧 Image Remove Background",

    "RemoveLatentMask+": "🔧 Remove Latent Mask",

    "ConditioningCombineMultiple+": "🔧 Conditionings Combine Multiple ",
    "ImageBatchMultiple+": "🔧 Images Batch Multiple",

    #"NoiseFromImage~": "🔧 Noise From Image",
}
