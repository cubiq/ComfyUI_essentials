import warnings
warnings.filterwarnings('ignore', module="torchvision")
import ast
import math
import random
import operator as op
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion

import torch
import torch.nn.functional as F

import torchvision.transforms.v2 as T

from nodes import MAX_RESOLUTION, SaveImage
import folder_paths
import comfy.utils

def p(image):
    return image.permute([0,3,1,2])
def pb(image):
    return image.permute([0,2,3,1])

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
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, "display": "number" }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, "display": "number" }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "keep_proportion": ("BOOLEAN", { "default": False }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, width, height, keep_proportion, interpolation="nearest"):
        if keep_proportion is True:
            _, oh, ow, _ = image.shape
            width = ow if width == 0 else width
            height = oh if height == 0 else height
            ratio = min(width / ow, height / oh)
            width = round(ow*ratio)
            height = round(oh*ratio)
        
        outputs = p(image)
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
                "width": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, "display": "number" }),
                "height": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, "display": "number" }),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"],),
                "x_offset": ("INT", { "default": 0, "min": -99999, "step": 1, "display": "number" }),
                "y_offset": ("INT", { "default": 0, "min": -99999, "step": 1, "display": "number" }),
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
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image):
        #image = p(image)
        #image = T.Grayscale(3)(image)
        #image = pb(image)
        #image = image.mean(dim=3, keepdim=True)
        #image = image.repeat(1, 1, 1, 3)
        image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        image = image.unsqueeze(-1).repeat(1, 1, 1, 3)
        return(image,)

class ImagePosterize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", { "default": 0.50, "min": 0.00, "max": 1.00, "step": 0.05, "display": "number" }),
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
                "exponent": ("FLOAT", { "default": 0.50, "min": 0.00, "max": 1.00, "step": 0.05, "display": "number" }),
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
            dim += (0,)
        if "x" in axis:
            dim += (1,)
        mask = torch.flip(mask, dim)

        return(mask,)

class MaskBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "size": ("INT", { "default": 5, "min": 1, "step": 1, "display": "number" }),
                "sigma": ("FLOAT", { "default": 1.0, "min": 0, "step": 0.5, "display": "number" }),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, mask, size, sigma):
        if size % 2 == 0:
            size+=1

        blurred = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        blurred = p(blurred)
        blurred = T.GaussianBlur(size, sigma)(blurred)
        blurred = pb(blurred)
        blurred = blurred[0, :, :, 0]

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
        results = self.save_images(preview, filename_prefix, prompt, extra_pnginfo)

        return( results )

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


NODE_CLASS_MAPPINGS = {
    "GetImageSize+": GetImageSize,

    "ImageResize+": ImageResize,
    "ImageCrop+": ImageCrop,
    "ImageFlip+": ImageFlip,

    "ImageDesaturate+": ImageDesaturate,
    "ImagePosterize+": ImagePosterize,
    "ImageCASharpening+": ImageCAS,
    "ImageEnhanceDifference+": ImageEnhanceDifference,

    "MaskBlur+": MaskBlur,
    "MaskFlip+": MaskFlip,
    "MaskPreview+": MaskPreview,

    "SimpleMath+": SimpleMath,
    "ConsoleDebug+": ConsoleDebug,

    "ModelCompile+": ModelCompile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GetImageSize+": "🔧 Get Image Size",

    "ImageResize+": "🔧 Image Resize",
    "ImageCrop+": "🔧 Image Crop",
    "ImageFlip+": "🔧 Image Flip",

    "ImageDesaturate+": "🔧 Image Desaturate",
    "ImagePosterize+": "🔧 Image Posterize",
    "ImageCASharpening+": "🔧 Image Contrast Adaptive Sharpening",
    "ImageEnhanceDifference+": "🔧 Image Enhance Difference",

    "MaskBlur+": "🔧 Mask Blur",
    "MaskFlip+": "🔧 Mask Flip",
    "MaskPreview+": "🔧 Mask Preview",

    "SimpleMath+": "🔧 Simple Math",
    "ConsoleDebug+": "🔧 Console Debug",

    "ModelCompile+": "🔧 Compile Model",
}
