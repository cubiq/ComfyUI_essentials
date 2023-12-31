import warnings
warnings.filterwarnings('ignore', module="torchvision")
import ast
import math
import random
import operator as op
import numpy as np

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
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
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

# highly experimental
def identify_keyframes_pytorch(frame_differences, threshold):
    keyframe_indices = []
    for i, diff in enumerate(frame_differences):
        if diff >= threshold:
            keyframe_indices.append(i + window_size)  # Adjusting index for window size
    return keyframe_indices

class ExtractKeyframes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "window_size": ("INT", { "default": 5, "min": 1, "step": 1, }),
                "max_frames": ("INT", { "default": 3, "min": 1, "step": 1, }),
                #"threshold": ("FLOAT", { "default": 0.75, "min": 0.00, "max": 1.00, "step": 0.05, }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, image, window_size, max_frames):
        mse_values = []

        for i in range(window_size, image.shape[0]):
            prev_frames_avg = torch.mean(image[i-window_size:i], dim=0)
            current_frame = image[i]
            mse = torch.mean((prev_frames_avg - current_frame) ** 2)
            mse_values.append(mse.item())

        sorted_mse = sorted(mse_values, reverse=True)
        if max_frames >= len(sorted_mse):
            threshold = sorted_mse[-1]
        else:
            threshold = sorted_mse[max_frames-1]
        
        print(threshold)

        keyframe_indices = []
        for i, diff in enumerate(mse_values):
            if diff >= threshold:
                keyframe_indices.append(i + window_size)

        out = image[keyframe_indices]
        print(keyframe_indices)
        return (out,)

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
        
        blurred = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 1)
        blurred = p(blurred)
        blurred = T.GaussianBlur(size, amount)(blurred)
        blurred = pb(blurred)
        blurred = blurred[:, :, :, 0]

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

from comfy_extras.nodes_stable3d import camera_embeddings
class StableZero123_Increments:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_vision": ("CLIP_VISION",),
                              "init_image": ("IMAGE",),
                              "vae": ("VAE",),
                              "width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "elevation_inc": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                              "azimuth_inc": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    FUNCTION = "encode"

    CATEGORY = "essentials"

    def encode(self, clip_vision, init_image, vae, width, height, batch_size, elevation, azimuth, elevation_inc, azimuth_inc):
        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1,1), width, height, "bilinear", "center").movedim(1,-1)
        encode_pixels = pixels[:,:,:,:3]
        t = vae.encode(encode_pixels)

        cam_embeds = [camera_embeddings(elevation + i * elevation_inc, azimuth + i * azimuth_inc) for i in range(batch_size)]
        cam_embeds = torch.cat(cam_embeds, dim=0)
        cond = torch.cat([pooled.repeat((batch_size, 1, 1)), cam_embeds], dim=-1)
       
        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return (positive, negative, {"samples":latent})

NODE_CLASS_MAPPINGS = {
    "StableZero123_Increments": StableZero123_Increments,

    "GetImageSize+": GetImageSize,

    "ImageResize+": ImageResize,
    "ImageCrop+": ImageCrop,
    "ImageFlip+": ImageFlip,

    "ImageDesaturate+": ImageDesaturate,
    "ImagePosterize+": ImagePosterize,
    "ImageCASharpening+": ImageCAS,
    "ImageEnhanceDifference+": ImageEnhanceDifference,
    "ImageExpandBatch+": ImageExpandBatch,
    "ImageFromBatch+": ImageFromBatch,
    "ExtractKeyframes+": ExtractKeyframes,

    "MaskBlur+": MaskBlur,
    "MaskFlip+": MaskFlip,
    "MaskPreview+": MaskPreview,
    "MaskBatch+": MaskBatch,
    "MaskExpandBatch+": MaskExpandBatch,
    "TransitionMask+": TransitionMask,
    "MaskFromColor+": MaskFromColor,
    "MaskFromBatch+": MaskFromBatch,

    "SimpleMath+": SimpleMath,
    "ConsoleDebug+": ConsoleDebug,

    "ModelCompile+": ModelCompile,
    "BatchCount+": BatchCount,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableZero123_Increments": "🔧 StableZero123 with Increments (temporary)",

    "GetImageSize+": "🔧 Get Image Size",
    "ImageResize+": "🔧 Image Resize",
    "ImageCrop+": "🔧 Image Crop",
    "ImageFlip+": "🔧 Image Flip",

    "ImageDesaturate+": "🔧 Image Desaturate",
    "ImagePosterize+": "🔧 Image Posterize",
    "ImageCASharpening+": "🔧 Image Contrast Adaptive Sharpening",
    "ImageEnhanceDifference+": "🔧 Image Enhance Difference",
    "ImageExpandBatch+": "🔧 Image Expand Batch",
    "ImageFromBatch+": "🔧 Image From Batch",
    "ExtractKeyframes+": "🔧 Extract Keyframes",

    "MaskBlur+": "🔧 Mask Blur",
    "MaskFlip+": "🔧 Mask Flip",
    "MaskPreview+": "🔧 Mask Preview",
    "MaskBatch+": "🔧 Mask Batch",
    "MaskExpandBatch+": "🔧 Mask Expand Batch",
    "TransitionMask+": "🔧 Transition Mask",
    "MaskFromColor+": "🔧 Mask From Color",
    "MaskFromBatch+": "🔧 MaskFromBatch",

    "SimpleMath+": "🔧 Simple Math",
    "ConsoleDebug+": "🔧 Console Debug",

    "ModelCompile+": "🔧 Compile Model",
    "BatchCount+": "🔧 Batch Count",
}
