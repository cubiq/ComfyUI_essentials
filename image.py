from .utils import max_, min_
from nodes import MAX_RESOLUTION
import comfy.utils

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

import warnings
warnings.filterwarnings('ignore', module="torchvision")
import math
import os
import numpy as np

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Image analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

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
    CATEGORY = "essentials/image analysis"

    def execute(self, image1, image2, exponent):
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.permute([0,3,1,2]), image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        diff_image = image1 - image2
        diff_image = torch.pow(diff_image, exponent)
        diff_image = torch.clamp(diff_image, 0, 1)

        return(diff_image,)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Batch tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

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
    CATEGORY = "essentials/image batch"

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
    CATEGORY = "essentials/image batch"

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
    CATEGORY = "essentials/image batch"

    def execute(self, image, start, length):
        if length<0:
            length = image.shape[0]
        start = min(start, image.shape[0]-1)
        length = min(image.shape[0]-start, length)
        return (image[start:start + length], )


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
    CATEGORY = "essentials/image batch"

    def execute(self, image):
        shape = image[0].shape[1:3]
        out = []

        for i in range(len(image)):
            img = image[i]
            if image[i].shape[1:3] != shape:
                img = comfy.utils.common_upscale(img.permute([0,3,1,2]), shape[1], shape[0], upscale_method='bicubic', crop='center').permute([0,2,3,1])
            out.append(img)

        out = torch.cat(out, dim=0)

        return (out,)


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Image manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

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
    CATEGORY = "essentials/image manipulation"

    def execute(self, image_from, image_to, mask):
        frames = mask.shape[0]

        if image_from.shape[1] != image_to.shape[1] or image_from.shape[2] != image_to.shape[2]:
            image_to = comfy.utils.common_upscale(image_to.permute([0,3,1,2]), image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        if frames < image_from.shape[0]:
            image_from = image_from[:frames]
        elif frames > image_from.shape[0]:
            image_from = torch.cat((image_from, image_from[-1].unsqueeze(0).repeat(frames-image_from.shape[0], 1, 1, 1)), dim=0)

        mask = mask.unsqueeze(3).repeat(1, 1, 1, 3)

        if image_from.shape[1] != mask.shape[1] or image_from.shape[2] != mask.shape[2]:
            mask = comfy.utils.common_upscale(mask.permute([0,3,1,2]), image_from.shape[2], image_from.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        out = mask * image_to + (1 - mask) * image_from

        return (out, )

class ImageResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "multiple_of": ("INT", { "default": 0, "min": 0, "max": 512, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if method == 'keep proportion' or method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = ow

            ratio = min(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)

            if method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith('fill'):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]

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
    CATEGORY = "essentials/image manipulation"

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
    CATEGORY = "essentials/image manipulation"

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

class ImageTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "cols": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "overlap": ("FLOAT", { "default": 0, "min": 0, "max": 0.5, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, rows, cols, overlap):
        h, w = image.shape[1:3]
        tile_h = h // rows
        tile_w = w // cols
        overlap_h = int(tile_h * overlap)
        overlap_w = int(tile_w * overlap)
        tile_h += overlap_h
        tile_w += overlap_w

        tiles = []
        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_h
                if j > 0:
                    x1 -= overlap_w

                y2 = y1 + tile_h
                x2 = x1 + tile_w

                if y2 > h:
                    y2 = h
                    y1 = y2 - tile_h
                if x2 > w:
                    x2 = w
                    x1 = x2 - tile_w

                tiles.append(image[:, y1:y2, x1:x2, :])

        tiles = torch.cat(tiles, dim=0)

        return(tiles,)

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
    CATEGORY = "essentials/image manipulation"
    FUNCTION = "execute"

    def execute(self, image, width, height, energy, order, keep_mask=None, drop_mask=None):
        from .carve import seam_carving

        img = image.permute([0, 3, 1, 2])

        if keep_mask is not None:
            #keep_mask = keep_mask.reshape((-1, 1, keep_mask.shape[-2], keep_mask.shape[-1])).movedim(1, -1)
            keep_mask = keep_mask.unsqueeze(1)

            if keep_mask.shape[2] != img.shape[2] or keep_mask.shape[3] != img.shape[3]:
                keep_mask = F.interpolate(keep_mask, size=(img.shape[2], img.shape[3]), mode="bilinear")
        if drop_mask is not None:
            drop_mask = drop_mask.unsqueeze(1)

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

        out = torch.stack(out).permute([0, 2, 3, 1])

        return(out, )

class ImageRandomTransform:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "repeat": ("INT", { "default": 1, "min": 1, "max": 256, "step": 1, }),
                "variation": ("FLOAT", { "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, seed, repeat, variation):
        h, w = image.shape[1:3]
        image = image.repeat(repeat, 1, 1, 1).permute([0, 3, 1, 2])

        distortion = 0.2 * variation
        rotation = 5 * variation
        brightness = 0.5 * variation
        contrast = 0.5 * variation
        saturation = 0.5 * variation
        hue = 0.2 * variation
        scale = 0.5 * variation

        torch.manual_seed(seed)

        out = []
        for i in image:
            tramsforms = T.Compose([
                T.RandomPerspective(distortion_scale=distortion, p=0.5),
                T.RandomRotation(degrees=rotation, interpolation=T.InterpolationMode.BILINEAR, expand=True),
                T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=(-hue, hue)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomResizedCrop((h, w), scale=(1-scale, 1+scale), ratio=(w/h, w/h), interpolation=T.InterpolationMode.BICUBIC),
            ])
            out.append(tramsforms(i.unsqueeze(0)))

        out = torch.cat(out, dim=0).permute([0, 2, 3, 1])

        return (out,)

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
    CATEGORY = "essentials/image manipulation"

    def execute(self, model, providers):
        from rembg import new_session

        model = model.split(":")[0]
        return (new_session(model, providers=[providers+"ExecutionProvider"]),)

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
    CATEGORY = "essentials/image manipulation"

    def execute(self, rembg_session, image):
        from rembg import remove as rembg

        image = image.permute([0, 3, 1, 2])
        output = []
        for img in image:
            img = T.ToPILImage()(img)
            img = rembg(img, session=rembg_session)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0)
        output = output.permute([0, 2, 3, 1])
        mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])

        return(output[:, :, :, :3], mask,)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Image processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class ImageDesaturate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", { "default": 1.00, "min": 0.00, "max": 1.00, "step": 0.05, }),
                "method": (["luminance (Rec.709)", "luminance (Rec.601)", "average", "lightness"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, factor, method):
        if method == "luminance (Rec.709)":
            grayscale = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        elif method == "luminance (Rec.601)":
            grayscale = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        elif method == "average":
            grayscale = image.mean(dim=3)
        elif method == "lightness":
            grayscale = (torch.max(image, dim=3)[0] + torch.min(image, dim=3)[0]) / 2

        grayscale = (1.0 - factor) * image + factor * grayscale.unsqueeze(-1).repeat(1, 1, 1, 3)
        grayscale = torch.clamp(grayscale, 0, 1)

        return(grayscale,)

class PixelOEPixelize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "downscale_mode": (["contrast", "bicubic", "nearest", "center", "k-centroid"],),
                "target_size": ("INT", { "default": 128, "min": 0, "max": MAX_RESOLUTION, "step": 8 }),
                "patch_size": ("INT", { "default": 16, "min": 4, "max": 32, "step": 2 }),
                "thickness": ("INT", { "default": 2, "min": 1, "max": 16, "step": 1 }),
                "color_matching": ("BOOLEAN", { "default": True }),
                "upscale": ("BOOLEAN", { "default": True }),
                #"contrast": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                #"saturation": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

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

        output = torch.stack(output, dim=0).permute([0, 2, 3, 1])

        return(output,)

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
    CATEGORY = "essentials/image processing"

    def execute(self, image, threshold):
        image = image.mean(dim=3, keepdim=True)
        image = (image > threshold).float()
        image = image.repeat(1, 1, 1, 3)

        return(image,)


LUTS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "luts")
# From https://github.com/yoonsikp/pycubelut/blob/master/pycubelut.py (MIT license)
class ImageApplyLUT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": ([f for f in os.listdir(LUTS_DIR) if f.lower().endswith('.cube')], ),
                "log_colorspace": ("BOOLEAN", { "default": False }),
                "clip_values": ("BOOLEAN", { "default": False }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1 }),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    # TODO: check if we can do without numpy
    def execute(self, image, lut_file, log_colorspace, clip_values, strength):
        from colour.io.luts.iridas_cube import read_LUT_IridasCube

        device = image.device
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
            lut_img = img.cpu().numpy().copy()

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

            lut_img = torch.from_numpy(lut_img).to(device)
            if strength < 1.0:
                lut_img = strength * lut_img + (1 - strength) * img
            out.append(lut_img)

        out = torch.stack(out)

        return (out, )

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
    CATEGORY = "essentials/image processing"
    FUNCTION = "execute"

    def execute(self, image, amount):
        epsilon = 1e-5
        img = F.pad(image.permute([0,3,1,2]), pad=(1, 1, 1, 1))

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
        inv_mx = torch.reciprocal(mx + epsilon)
        amp = inv_mx * torch.minimum(mn, (2 - mx))

        # scaling
        amp = torch.sqrt(amp)
        w = - amp * (amount * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4*w)

        output = ((b + d + f + h)*w + e) * div
        output = output.clamp(0, 1)
        #output = torch.nan_to_num(output)

        output = output.permute([0,2,3,1])

        return (output,)

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


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class ImageToDevice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image, device):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        image = image.to(device)
        torch.cuda.empty_cache()

        return (image,)

class GetImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT",)
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image):
        return (image.shape[2], image.shape[1], image.shape[0])

class ImageRemoveAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image):
        if image.shape[3] == 4:
            image = image[..., :3]
        return (image,)

IMAGE_CLASS_MAPPINGS = {
    # Image analysis
    "ImageEnhanceDifference+": ImageEnhanceDifference,

    # Image batch
    "ImageBatchMultiple+": ImageBatchMultiple,
    "ImageExpandBatch+": ImageExpandBatch,
    #"ImageFromBatch+": ImageFromBatch,
    "ImageListToBatch+": ImageListToBatch,

    # Image manipulation
    "ImageCompositeFromMaskBatch+": ImageCompositeFromMaskBatch,
    "ImageCrop+": ImageCrop,
    "ImageFlip+": ImageFlip,
    "ImageRandomTransform+": ImageRandomTransform,
    "ImageRemoveAlpha+": ImageRemoveAlpha,
    "ImageRemoveBackground+": ImageRemoveBackground,
    "ImageResize+": ImageResize,
    "ImageSeamCarving+": ImageSeamCarving,
    "ImageTile+": ImageTile,
    "RemBGSession+": RemBGSession,

    # Image processing
    "ImageApplyLUT+": ImageApplyLUT,
    "ImageCASharpening+": ImageCAS,
    "ImageDesaturate+": ImageDesaturate,
    "PixelOEPixelize+": PixelOEPixelize,
    "ImagePosterize+": ImagePosterize,

    # Utilities
    "GetImageSize+": GetImageSize,
    "ImageToDevice+": ImageToDevice,

    #"ExtractKeyframes+": ExtractKeyframes,
}

IMAGE_NAME_MAPPINGS = {
    # Image analysis
    "ImageEnhanceDifference+": "🔧 Image Enhance Difference",

    # Image batch
    "ImageBatchMultiple+": "🔧 Images Batch Multiple",
    "ImageExpandBatch+": "🔧 Image Expand Batch",
    #"ImageFromBatch+": "🔧 Image From Batch",
    "ImageListToBatch+": "🔧 Image List To Batch",

    # Image manipulation
    "ImageCompositeFromMaskBatch+": "🔧 Image Composite From Mask Batch",
    "ImageCrop+": "🔧 Image Crop",
    "ImageFlip+": "🔧 Image Flip",
    "ImageRandomTransform+": "🔧 Image Random Transform",
    "ImageRemoveAlpha+": "🔧 Image Remove Alpha",
    "ImageRemoveBackground+": "🔧 Image Remove Background",
    "ImageResize+": "🔧 Image Resize",
    "ImageSeamCarving+": "🔧 Image Seam Carving",
    "ImageTile+": "🔧 Image Tile",
    "RemBGSession+": "🔧 RemBG Session",

    # Image processing
    "ImageApplyLUT+": "🔧 Image Apply LUT",
    "ImageCASharpening+": "🔧 Image Contrast Adaptive Sharpening",
    "ImageDesaturate+": "🔧 Image Desaturate",
    "PixelOEPixelize+": "🔧 Pixelize",
    "ImagePosterize+": "🔧 Image Posterize",

    # Utilities
    "GetImageSize+": "🔧 Get Image Size",
    "ImageToDevice+": "🔧 Image To Device",
}
