from nodes import SaveImage
import torch
import torchvision.transforms.v2 as T
import random
import folder_paths
import comfy.utils
from .image import ImageExpandBatch
from .utils import AnyType
import numpy as np
import scipy
from PIL import Image
from nodes import MAX_RESOLUTION
import math

any = AnyType("*")

class MaskBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", { "default": 6, "min": 0, "max": 256, "step": 1, }),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, amount, device):
        if amount == 0:
            return (mask,)

        if "gpu" == device:
            mask = mask.to(comfy.model_management.get_torch_device())
        elif "cpu" == device:
            mask = mask.to('cpu')

        if amount % 2 == 0:
            amount+= 1

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1)

        if "gpu" == device or "cpu" == device:
            mask = mask.to(comfy.model_management.intermediate_device())

        return(mask,)

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
    CATEGORY = "essentials/mask"

    def execute(self, mask, axis):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        dim = ()
        if "y" in axis:
            dim += (1,)
        if "x" in axis:
            dim += (2,)
        mask = torch.flip(mask, dims=dim)

        return(mask,)

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
    CATEGORY = "essentials/mask"

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
    CATEGORY = "essentials/mask batch"

    def execute(self, mask1, mask2):
        if mask1.shape[1:] != mask2.shape[1:]:
            mask2 = comfy.utils.common_upscale(mask2.unsqueeze(1).expand(-1,3,-1,-1), mask1.shape[2], mask1.shape[1], upscale_method='bicubic', crop='center')[:,0,:,:]

        return (torch.cat((mask1, mask2), dim=0),)

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
    CATEGORY = "essentials/mask batch"

    def execute(self, mask, size, method):
        return (ImageExpandBatch().execute(mask.unsqueeze(1).expand(-1,3,-1,-1), size, method)[0][:,0,:,:],)


class MaskBoundingBox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", { "default": 0, "min": 0, "max": 4096, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("MASK", "IMAGE", "x", "y", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, padding, blur, image_optional=None):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        if image_optional is None:
            image_optional = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        # resize the image if it's not the same size as the mask
        if image_optional.shape[1:] != mask.shape[1:]:
            image_optional = comfy.utils.common_upscale(image_optional.permute([0,3,1,2]), mask.shape[2], mask.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

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
    CATEGORY = "essentials/mask"

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
    CATEGORY = "essentials/mask"

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
                mask = torch.from_numpy(scipy.ndimage.binary_opening(mask.cpu().numpy(), structure=np.ones((remove_isolated_pixels, remove_isolated_pixels))))

            # fill holes
            if fill_holes:
                mask = torch.from_numpy(scipy.ndimage.binary_fill_holes(mask.cpu().numpy()))

            # if the mask is too small, it's probably noise
            if mask.sum() / (mask.shape[0]*mask.shape[1]) > remove_small_masks:
                masks.append(mask)

        if masks == []:
            masks.append(torch.zeros_like(im)[:,:,0]) # return an empty mask if no masks were found, prevents errors

        mask = torch.stack(masks, dim=0).float()

        return (mask, )


class MaskFix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "erode_dilate": ("INT", { "default": 0, "min": -256, "max": 256, "step": 1, }),
                "fill_holes": ("INT", { "default": 0, "min": 0, "max": 128, "step": 1, }),
                "remove_isolated_pixels": ("INT", { "default": 0, "min": 0, "max": 32, "step": 1, }),
                "smooth": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 256, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, erode_dilate, smooth, remove_isolated_pixels, blur, fill_holes):
        masks = []
        for m in mask:
            # erode and dilate
            if erode_dilate != 0:
                if erode_dilate < 0:
                    m = torch.from_numpy(scipy.ndimage.grey_erosion(m.cpu().numpy(), size=(-erode_dilate, -erode_dilate)))
                else:
                    m = torch.from_numpy(scipy.ndimage.grey_dilation(m.cpu().numpy(), size=(erode_dilate, erode_dilate)))

            # fill holes
            if fill_holes > 0:
                #m = torch.from_numpy(scipy.ndimage.binary_fill_holes(m.cpu().numpy(), structure=np.ones((fill_holes,fill_holes)))).float()
                m = torch.from_numpy(scipy.ndimage.grey_closing(m.cpu().numpy(), size=(fill_holes, fill_holes)))

            # remove isolated pixels
            if remove_isolated_pixels > 0:
                m = torch.from_numpy(scipy.ndimage.grey_opening(m.cpu().numpy(), size=(remove_isolated_pixels, remove_isolated_pixels)))

            # smooth the mask
            if smooth > 0:
                if smooth % 2 == 0:
                    smooth += 1
                m = T.functional.gaussian_blur((m > 0.5).unsqueeze(0), smooth).squeeze(0)

            # blur the mask
            if blur > 0:
                if blur % 2 == 0:
                    blur += 1
                m = T.functional.gaussian_blur(m.float().unsqueeze(0), blur).squeeze(0)

            masks.append(m.float())

        masks = torch.stack(masks, dim=0).float()

        return (masks, )

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
    CATEGORY = "essentials/mask"

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
                "length": ("INT", { "default": 1, "min": 1, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask batch"

    def execute(self, mask, start, length):
        if length > mask.shape[0]:
            length = mask.shape[0]

        start = min(start, mask.shape[0]-1)
        length = min(mask.shape[0]-start, length)
        return (mask[start:start + length], )

class MaskFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", { "default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
            }, "optional": {
                "values": (any, { "default": 0.0, "min": 0.0, "max": 1.0, }),
                "str_values": ("STRING", { "default": "", "multiline": True, "placeholder": "0.0, 0.5, 1.0",}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, width, height, values=None, str_values=""):
        out = []

        if values is not None:
            if not isinstance(values, list):
                out = [values]
            else:
                out.extend([float(v) for v in values])

        if str_values != "":
            str_values = [float(v) for v in str_values.split(",")]
            out.extend(str_values)

        if out == []:
            raise ValueError("No values provided")
        
        out = torch.tensor(out).float().clamp(0.0, 1.0)
        out = out.view(-1, 1, 1).expand(-1, height, width)
        
        values = None
        str_values = ""

        return (out, )

class MaskFromRGBCMYBW:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "threshold_r": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_g": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_b": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, image, threshold_r, threshold_g, threshold_b):
        red = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)).float()
        green = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
        blue = ((image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1-threshold_b)).float()

        cyan = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        magenta = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1-threshold_b)).float()
        yellow = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()

        black = ((image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)).float()
        white = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        
        return (red, green, blue, cyan, magenta, yellow, black, white,)

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
    CATEGORY = "essentials/mask"

    def linear(self, i, t):
        return i/t
    def ease_in(self, i, t):
        return pow(i/t, 2)
    def ease_out(self, i, t):
        return 1 - pow(1 - i/t, 2)
    def ease_in_out(self, i, t):
        if i < t/2:
            return pow(i/(t/2), 2) / 2
        else:
            return 1 - pow(1 - (i - t/2)/(t/2), 2) / 2

    def execute(self, width, height, frames, start_frame, end_frame, transition_type, timing_function):
        if timing_function == 'in':
            timing_function = self.ease_in
        elif timing_function == 'out':
            timing_function = self.ease_out
        elif timing_function == 'in-out':
            timing_function = self.ease_in_out
        else:
            timing_function = self.linear

        out = []

        end_frame = min(frames, end_frame)
        transition = end_frame - start_frame

        if start_frame > 0:
            out = out + [torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")] * start_frame

        for i in range(transition):
            frame = torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")
            progress = timing_function(i, transition-1)

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

MASK_CLASS_MAPPINGS = {
    "MaskBlur+": MaskBlur,
    "MaskBoundingBox+": MaskBoundingBox,
    "MaskFix+": MaskFix,
    "MaskFlip+": MaskFlip,
    "MaskFromColor+": MaskFromColor,
    "MaskFromList+": MaskFromList,
    "MaskFromRGBCMYBW+": MaskFromRGBCMYBW,
    "MaskFromSegmentation+": MaskFromSegmentation,
    "MaskPreview+": MaskPreview,
    "MaskSmooth+": MaskSmooth,
    "TransitionMask+": TransitionMask,

    # Batch
    "MaskBatch+": MaskBatch,
    "MaskExpandBatch+": MaskExpandBatch,
    "MaskFromBatch+": MaskFromBatch,
}

MASK_NAME_MAPPINGS = {
    "MaskBlur+": "🔧 Mask Blur",
    "MaskFix+": "🔧 Mask Fix",
    "MaskFlip+": "🔧 Mask Flip",
    "MaskFromColor+": "🔧 Mask From Color",
    "MaskFromList+": "🔧 Mask From List",
    "MaskFromRGBCMYBW+": "🔧 Mask From RGB/CMY/BW",
    "MaskFromSegmentation+": "🔧 Mask From Segmentation",
    "MaskPreview+": "🔧 Mask Preview",
    "MaskBoundingBox+": "🔧 Mask Bounding Box",
    "MaskSmooth+": "🔧 Mask Smooth",
    "TransitionMask+": "🔧 Transition Mask",

    "MaskBatch+": "🔧 Mask Batch",
    "MaskExpandBatch+": "🔧 Mask Expand Batch",
    "MaskFromBatch+": "🔧 Mask From Batch",
}
