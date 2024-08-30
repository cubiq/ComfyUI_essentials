from .utils import max_, min_
from nodes import MAX_RESOLUTION
import comfy.utils
from nodes import SaveImage
from node_helpers import pillow
from PIL import Image, ImageOps

import kornia
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

#import warnings
#warnings.filterwarnings('ignore', module="torchvision")
import math
import os
import numpy as np
import folder_paths
from pathlib import Path
import random

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
                "method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], { "default": "lanczos" }),
            }, "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image_1, method, image_2=None, image_3=None, image_4=None, image_5=None):
        out = image_1

        if image_2 is not None:
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

class ImageBatchToList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image):
        return ([image[i].unsqueeze(0) for i in range(image.shape[0])], )


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

class ImageComposite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "y": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "offset_x": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "offset_y": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, destination, source, x, y, offset_x, offset_y, mask=None):
        if mask is None:
            mask = torch.ones_like(source)[:,:,:,0]
        
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)

        if mask.shape[1:3] != source.shape[1:3]:
            mask = F.interpolate(mask.permute([0, 3, 1, 2]), size=(source.shape[1], source.shape[2]), mode='bicubic')
            mask = mask.permute([0, 2, 3, 1])
        
        if mask.shape[0] > source.shape[0]:
            mask = mask[:source.shape[0]]
        elif mask.shape[0] < source.shape[0]:
            mask = torch.cat((mask, mask[-1:].repeat((source.shape[0]-mask.shape[0], 1, 1, 1))), dim=0)
        
        if destination.shape[0] > source.shape[0]:
            destination = destination[:source.shape[0]]
        elif destination.shape[0] < source.shape[0]:
            destination = torch.cat((destination, destination[-1:].repeat((source.shape[0]-destination.shape[0], 1, 1, 1))), dim=0)
        
        if not isinstance(x, list):
            x = [x]
        if not isinstance(y, list):
            y = [y]
        
        if len(x) < destination.shape[0]:
            x = x + [x[-1]] * (destination.shape[0] - len(x))
        if len(y) < destination.shape[0]:
            y = y + [y[-1]] * (destination.shape[0] - len(y))
        
        x = [i + offset_x for i in x]
        y = [i + offset_y for i in y]

        output = []
        for i in range(destination.shape[0]):
            d = destination[i].clone()
            s = source[i]
            m = mask[i]

            if x[i]+source.shape[2] > destination.shape[2]:
                s = s[:, :, :destination.shape[2]-x[i], :]
                m = m[:, :, :destination.shape[2]-x[i], :]
            if y[i]+source.shape[1] > destination.shape[1]:
                s = s[:, :destination.shape[1]-y[i], :, :]
                m = m[:destination.shape[1]-y[i], :, :]
            
            #output.append(s * m + d[y[i]:y[i]+s.shape[0], x[i]:x[i]+s.shape[1], :] * (1 - m))
            d[y[i]:y[i]+s.shape[0], x[i]:x[i]+s.shape[1], :] = s * m + d[y[i]:y[i]+s.shape[0], x[i]:x[i]+s.shape[1], :] * (1 - m)
            output.append(d)
        
        output = torch.stack(output)

        # apply the source to the destination at XY position using the mask
        #for i in range(destination.shape[0]):
        #    output[i, y[i]:y[i]+source.shape[1], x[i]:x[i]+source.shape[2], :] = source * mask + destination[i, y[i]:y[i]+source.shape[1], x[i]:x[i]+source.shape[2], :] * (1 - mask)

        #for x_, y_ in zip(x, y):
        #    output[:, y_:y_+source.shape[1], x_:x_+source.shape[2], :] = source * mask + destination[:, y_:y_+source.shape[1], x_:x_+source.shape[2], :] * (1 - mask)

        #output[:, y:y+source.shape[1], x:x+source.shape[2], :] = source * mask + destination[:, y:y+source.shape[1], x:x+source.shape[2], :] * (1 - mask)
        #output = destination * (1 - mask) + source * mask

        return (output,)

class ImageResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1, }),
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
                height = oh

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
        
        outputs = torch.clamp(outputs, 0, 1)

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
                "overlap_x": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
                "overlap_y": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "tile_width", "tile_height", "overlap_x", "overlap_y",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, rows, cols, overlap, overlap_x, overlap_y):
        h, w = image.shape[1:3]
        tile_h = h // rows
        tile_w = w // cols
        h = tile_h * rows
        w = tile_w * cols
        overlap_h = int(tile_h * overlap) + overlap_y
        overlap_w = int(tile_w * overlap) + overlap_x

        # max overlap is half of the tile size
        overlap_h = min(tile_h // 2, overlap_h)
        overlap_w = min(tile_w // 2, overlap_w)

        if rows == 1:
            overlap_h = 0
        if cols == 1:
            overlap_w = 0
        
        tiles = []
        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_h
                if j > 0:
                    x1 -= overlap_w

                y2 = y1 + tile_h + overlap_h
                x2 = x1 + tile_w + overlap_w

                if y2 > h:
                    y2 = h
                    y1 = y2 - tile_h - overlap_h
                if x2 > w:
                    x2 = w
                    x1 = x2 - tile_w - overlap_w

                tiles.append(image[:, y1:y2, x1:x2, :])
        tiles = torch.cat(tiles, dim=0)

        return(tiles, tile_w+overlap_w, tile_h+overlap_h, overlap_w, overlap_h,)

class ImageUntile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "overlap_x": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
                "overlap_y": ("INT", { "default": 0, "min": 0, "max": MAX_RESOLUTION//2, "step": 1, }),
                "rows": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "cols": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, tiles, overlap_x, overlap_y, rows, cols):
        tile_h, tile_w = tiles.shape[1:3]
        tile_h -= overlap_y
        tile_w -= overlap_x
        out_w = cols * tile_w
        out_h = rows * tile_h

        out = torch.zeros((1, out_h, out_w, tiles.shape[3]), device=tiles.device, dtype=tiles.dtype)

        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_y
                if j > 0:
                    x1 -= overlap_x

                y2 = y1 + tile_h + overlap_y
                x2 = x1 + tile_w + overlap_x

                if y2 > out_h:
                    y2 = out_h
                    y1 = y2 - tile_h - overlap_y
                if x2 > out_w:
                    x2 = out_w
                    x1 = x2 - tile_w - overlap_x
                
                mask = torch.ones((1, tile_h+overlap_y, tile_w+overlap_x), device=tiles.device, dtype=tiles.dtype)

                # feather the overlap on top
                if i > 0 and overlap_y > 0:
                    mask[:, :overlap_y, :] *= torch.linspace(0, 1, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
                # feather the overlap on bottom
                #if i < rows - 1:
                #    mask[:, -overlap_y:, :] *= torch.linspace(1, 0, overlap_y, device=tiles.device, dtype=tiles.dtype).unsqueeze(1)
                # feather the overlap on left
                if j > 0 and overlap_x > 0:
                    mask[:, :, :overlap_x] *= torch.linspace(0, 1, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
                # feather the overlap on right
                #if j < cols - 1:
                #    mask[:, :, -overlap_x:] *= torch.linspace(1, 0, overlap_x, device=tiles.device, dtype=tiles.dtype).unsqueeze(0)
                
                mask = mask.unsqueeze(-1).repeat(1, 1, 1, tiles.shape[3])
                tile = tiles[i * cols + j] * mask
                out[:, y1:y2, x1:x2, :] = out[:, y1:y2, x1:x2, :] * (1 - mask) + tile
        return(out, )

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

        out = torch.cat(out, dim=0).permute([0, 2, 3, 1]).clamp(0, 1)

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
        from rembg import new_session, remove

        model = model.split(":")[0]

        class Session:
            def __init__(self, model, providers):
                self.session = new_session(model, providers=[providers+"ExecutionProvider"])
            def process(self, image):
                return remove(image, session=self.session)
            
        return (Session(model, providers),)

class TransparentBGSession:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["base", "fast", "base-nightly"],),
                "use_jit": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("REMBG_SESSION",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, mode, use_jit):
        from transparent_background import Remover

        class Session:
            def __init__(self, mode, use_jit):
                self.session = Remover(mode=mode, jit=use_jit)
            def process(self, image):
                return self.session.process(image)

        return (Session(mode, use_jit),)

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
        image = image.permute([0, 3, 1, 2])
        output = []
        for img in image:
            img = T.ToPILImage()(img)
            img = rembg_session.process(img)
            output.append(T.ToTensor()(img))

        output = torch.stack(output, dim=0)
        output = output.permute([0, 2, 3, 1])
        mask = output[:, :, :, 3] if output.shape[3] == 4 else torch.ones_like(output[:, :, :, 0])
        # output = output[:, :, :, :3]

        return(output, mask,)

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

# From https://github.com/yoonsikp/pycubelut/blob/master/pycubelut.py (MIT license)
class ImageApplyLUT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": (folder_paths.get_filename_list("luts"),),
                "gamma_correction": ("BOOLEAN", { "default": True }),
                "clip_values": ("BOOLEAN", { "default": True }),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1 }),
            }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    # TODO: check if we can do without numpy
    def execute(self, image, lut_file, gamma_correction, clip_values, strength):
        lut_file_path = folder_paths.get_full_path("luts", lut_file)
        if not lut_file_path or not Path(lut_file_path).exists():
            print(f"Could not find LUT file: {lut_file_path}")
            return (image,)
            
        from colour.io.luts.iridas_cube import read_LUT_IridasCube
        
        device = image.device
        lut = read_LUT_IridasCube(lut_file_path)
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
            if gamma_correction:
                lut_img = lut_img ** (1/2.2)
            lut_img = lut.apply(lut_img)
            if gamma_correction:
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

class ImageSmartSharpen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_radius": ("INT", { "default": 7, "min": 1, "max": 25, "step": 1, }),
                "preserve_edges": ("FLOAT", { "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "sharpen": ("FLOAT", { "default": 5.0, "min": 0.0, "max": 25.0, "step": 0.5 }),
                "ratio": ("FLOAT", { "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1 }),
        }}

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials/image processing"
    FUNCTION = "execute"

    def execute(self, image, noise_radius, preserve_edges, sharpen, ratio):
        import cv2

        output = []
        #diagonal = np.sqrt(image.shape[1]**2 + image.shape[2]**2)
        if preserve_edges > 0:
            preserve_edges = max(1 - preserve_edges, 0.05)

        for img in image:
            if noise_radius > 1:
                sigma = 0.3 * ((noise_radius - 1) * 0.5 - 1) + 0.8 # this is what pytorch uses for blur
                #sigma_color = preserve_edges * (diagonal / 2048)
                blurred = cv2.bilateralFilter(img.cpu().numpy(), noise_radius, preserve_edges, sigma)
                blurred = torch.from_numpy(blurred)
            else:
                blurred = img

            if sharpen > 0:
                sharpened = kornia.enhance.sharpness(img.permute(2,0,1), sharpen).permute(1,2,0)
            else:
                sharpened = img

            img = ratio * sharpened + (1 - ratio) * blurred
            img = torch.clamp(img, 0, 1)
            output.append(img)
        
        del blurred, sharpened
        output = torch.stack(output)

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

class ImageColorMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_space": (["LAB", "YCbCr", "RGB", "LUV", "YUV", "XYZ"],),
                "factor": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, }),
                "device": (["auto", "cpu", "gpu"],),
                "batch_size": ("INT", { "default": 0, "min": 0, "max": 1024, "step": 1, }),
            },
            "optional": {
                "reference_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, reference, color_space, factor, device, batch_size, reference_mask=None):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        image = image.permute([0, 3, 1, 2])
        reference = reference.permute([0, 3, 1, 2]).to(device)
         
        # Ensure reference_mask is in the correct format and on the right device
        if reference_mask is not None:
            assert reference_mask.ndim == 3, f"Expected reference_mask to have 3 dimensions, but got {reference_mask.ndim}"
            assert reference_mask.shape[0] == reference.shape[0], f"Frame count mismatch: reference_mask has {reference_mask.shape[0]} frames, but reference has {reference.shape[0]}"
            
            # Reshape mask to (batch, 1, height, width)
            reference_mask = reference_mask.unsqueeze(1).to(device)
             
            # Ensure the mask is binary (0 or 1)
            reference_mask = (reference_mask > 0.5).float()
             
            # Ensure spatial dimensions match
            if reference_mask.shape[2:] != reference.shape[2:]:
                reference_mask = comfy.utils.common_upscale(
                    reference_mask,
                    reference.shape[3], reference.shape[2],
                    upscale_method='bicubic',
                    crop='center'
                )

        if batch_size == 0 or batch_size > image.shape[0]:
            batch_size = image.shape[0]

        if "LAB" == color_space:
            reference = kornia.color.rgb_to_lab(reference)
        elif "YCbCr" == color_space:
            reference = kornia.color.rgb_to_ycbcr(reference)
        elif "LUV" == color_space:
            reference = kornia.color.rgb_to_luv(reference)
        elif "YUV" == color_space:
            reference = kornia.color.rgb_to_yuv(reference)
        elif "XYZ" == color_space:
            reference = kornia.color.rgb_to_xyz(reference)

        reference_mean, reference_std = self.compute_mean_std(reference, reference_mask)

        image_batch = torch.split(image, batch_size, dim=0)
        output = []

        for image in image_batch:
            image = image.to(device)

            if color_space == "LAB":
                image = kornia.color.rgb_to_lab(image)
            elif color_space == "YCbCr":
                image = kornia.color.rgb_to_ycbcr(image)
            elif color_space == "LUV":
                image = kornia.color.rgb_to_luv(image)
            elif color_space == "YUV":
                image = kornia.color.rgb_to_yuv(image)
            elif color_space == "XYZ":
                image = kornia.color.rgb_to_xyz(image)

            image_mean, image_std = self.compute_mean_std(image)

            matched = torch.nan_to_num((image - image_mean) / image_std) * torch.nan_to_num(reference_std) + reference_mean
            matched = factor * matched + (1 - factor) * image

            if color_space == "LAB":
                matched = kornia.color.lab_to_rgb(matched)
            elif color_space == "YCbCr":
                matched = kornia.color.ycbcr_to_rgb(matched)
            elif color_space == "LUV":
                matched = kornia.color.luv_to_rgb(matched)
            elif color_space == "YUV":
                matched = kornia.color.yuv_to_rgb(matched)
            elif color_space == "XYZ":
                matched = kornia.color.xyz_to_rgb(matched)

            out = matched.permute([0, 2, 3, 1]).clamp(0, 1).to(comfy.model_management.intermediate_device())
            output.append(out)

        out = None
        output = torch.cat(output, dim=0)
        return (output,)

    def compute_mean_std(self, tensor, mask=None):
        if mask is not None:
            # Apply mask to the tensor
            masked_tensor = tensor * mask

            # Calculate the sum of the mask for each channel
            mask_sum = mask.sum(dim=[2, 3], keepdim=True)

            # Avoid division by zero
            mask_sum = torch.clamp(mask_sum, min=1e-6)

            # Calculate mean and std only for masked area
            mean = torch.nan_to_num(masked_tensor.sum(dim=[2, 3], keepdim=True) / mask_sum)
            std = torch.sqrt(torch.nan_to_num(((masked_tensor - mean) ** 2 * mask).sum(dim=[2, 3], keepdim=True) / mask_sum))
        else:
            mean = tensor.mean(dim=[2, 3], keepdim=True)
            std = tensor.std(dim=[2, 3], keepdim=True)
        return mean, std

class ImageColorMatchAdobe(ImageColorMatch):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "color_space": (["RGB", "LAB"],),
                "luminance_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "color_intensity_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "fade_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "neutralization_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "device": (["auto", "cpu", "gpu"],),
            },
            "optional": {
                "reference_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def analyze_color_statistics(self, image, mask=None):
        # Assuming image is in RGB format
        l, a, b = kornia.color.rgb_to_lab(image).chunk(3, dim=1)

        if mask is not None:
            # Ensure mask is binary and has the same spatial dimensions as the image
            mask = F.interpolate(mask, size=image.shape[2:], mode='nearest')
            mask = (mask > 0.5).float()
            
            # Apply mask to each channel
            l = l * mask
            a = a * mask
            b = b * mask
            
            # Compute masked mean and std
            num_pixels = mask.sum()
            mean_l = (l * mask).sum() / num_pixels
            mean_a = (a * mask).sum() / num_pixels
            mean_b = (b * mask).sum() / num_pixels
            std_l = torch.sqrt(((l - mean_l)**2 * mask).sum() / num_pixels)
            var_ab = ((a - mean_a)**2 + (b - mean_b)**2) * mask
            std_ab = torch.sqrt(var_ab.sum() / num_pixels)
        else:
            mean_l = l.mean()
            std_l = l.std()
            mean_a = a.mean()
            mean_b = b.mean()
            std_ab = torch.sqrt(a.var() + b.var())

        return mean_l, std_l, mean_a, mean_b, std_ab

    def apply_color_transformation(self, image, source_stats, dest_stats, L, C, N):
        l, a, b = kornia.color.rgb_to_lab(image).chunk(3, dim=1)
        
        # Unpack statistics
        src_mean_l, src_std_l, src_mean_a, src_mean_b, src_std_ab = source_stats
        dest_mean_l, dest_std_l, dest_mean_a, dest_mean_b, dest_std_ab = dest_stats

        # Adjust luminance
        l_new = (l - dest_mean_l) * (src_std_l / dest_std_l) * L + src_mean_l

        # Neutralize color cast
        a = a - N * dest_mean_a
        b = b - N * dest_mean_b

        # Adjust color intensity
        a_new = a * (src_std_ab / dest_std_ab) * C
        b_new = b * (src_std_ab / dest_std_ab) * C

        # Combine channels
        lab_new = torch.cat([l_new, a_new, b_new], dim=1)

        # Convert back to RGB
        rgb_new = kornia.color.lab_to_rgb(lab_new)

        return rgb_new

    def execute(self, image, reference, color_space, luminance_factor, color_intensity_factor, fade_factor, neutralization_factor, device, reference_mask=None):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        # Ensure image and reference are in the correct shape (B, C, H, W)
        image = image.permute(0, 3, 1, 2).to(device)
        reference = reference.permute(0, 3, 1, 2).to(device)

        # Handle reference_mask (if provided)
        if reference_mask is not None:
            # Ensure reference_mask is 4D (B, 1, H, W)
            if reference_mask.ndim == 2:
                reference_mask = reference_mask.unsqueeze(0).unsqueeze(0)
            elif reference_mask.ndim == 3:
                reference_mask = reference_mask.unsqueeze(1)
            reference_mask = reference_mask.to(device)

         # Analyze color statistics
        source_stats = self.analyze_color_statistics(reference, reference_mask)
        dest_stats = self.analyze_color_statistics(image)

        # Apply color transformation
        transformed = self.apply_color_transformation(
            image, source_stats, dest_stats, 
            luminance_factor, color_intensity_factor, neutralization_factor
        )

        # Apply fade factor
        result = fade_factor * transformed + (1 - fade_factor) * image

        # Convert back to (B, H, W, C) format and ensure values are in [0, 1] range
        result = result.permute(0, 2, 3, 1).clamp(0, 1).to(comfy.model_management.intermediate_device())

        return (result,)


class ImageHistogramMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "reference": ("IMAGE",),
                "method": (["pytorch", "skimage"],),
                "factor": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, }),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, reference, method, factor, device):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        if "pytorch" in method:
            from .histogram_matching import Histogram_Matching

            image = image.permute([0, 3, 1, 2]).to(device)
            reference = reference.permute([0, 3, 1, 2]).to(device)[0].unsqueeze(0)
            image.requires_grad = True
            reference.requires_grad = True

            out = []

            for i in image:
                i = i.unsqueeze(0)
                hm = Histogram_Matching(differentiable=True)
                out.append(hm(i, reference))
            out = torch.cat(out, dim=0)
            out = factor * out + (1 - factor) * image
            out = out.permute([0, 2, 3, 1]).clamp(0, 1)
        else:
            from skimage.exposure import match_histograms

            out = torch.from_numpy(match_histograms(image.cpu().numpy(), reference.cpu().numpy(), channel_axis=3)).to(device)
            out = factor * out + (1 - factor) * image.to(device)

        return (out.to(comfy.model_management.intermediate_device()),)

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

        image = image.clone().to(device)
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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image):
        if image.shape[3] == 4:
            image = image[..., :3]
        return (image,)

class ImagePreviewFromLatent(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE", ),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64})
            }, "optional": {
                "image": (["none"], {"image_upload": False}),
            }, "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, latent, vae, tile_size, prompt=None, extra_pnginfo=None, image=None, filename_prefix="ComfyUI"):
        mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        ui = None

        if image.startswith("clipspace"):
            image_path = folder_paths.get_annotated_filepath(image)
            if not os.path.exists(image_path):
                raise ValueError(f"Clipspace image does not exist anymore, select 'none' in the image field.")

            img = pillow(Image.open, image_path)
            img = pillow(ImageOps.exif_transpose, img)
            if img.mode == "I":
                img = img.point(lambda i: i * (1 / 255))
            image = img.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            ui = {
                "filename": os.path.basename(image_path),
                "subfolder": os.path.dirname(image_path),
                "type": "temp",
            }
        else:
            if tile_size > 0:
                tile_size = max(tile_size, 320)
                image = vae.decode_tiled(latent["samples"], tile_x=tile_size // 8, tile_y=tile_size // 8, )
            else:
                image = vae.decode(latent["samples"])
            ui = self.save_images(image, filename_prefix, prompt, extra_pnginfo)

        out = {**ui, "result": (image, mask, image.shape[2], image.shape[1],)}
        return out

class NoiseFromImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_strenght": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "noise_size": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "color_noise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_scale_diff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "saturation": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1 }),
            },
            "optional": {
                "noise_mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image, noise_size, color_noise, mask_strength, mask_scale_diff, mask_contrast, noise_strenght, saturation, contrast, blur, noise_mask=None):
        torch.manual_seed(0)

        elastic_alpha = max(image.shape[1], image.shape[2])# * noise_size
        elastic_sigma = elastic_alpha / 400 * noise_size

        blur_size = int(6 * blur+1)
        if blur_size % 2 == 0:
            blur_size+= 1

        if noise_mask is None:
            noise_mask = image
        
        # increase contrast of the mask
        if mask_contrast != 1:
            noise_mask = T.ColorJitter(contrast=(mask_contrast,mask_contrast))(noise_mask.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])

        # Ensure noise mask is the same size as the image
        if noise_mask.shape[1:] != image.shape[1:]:
            noise_mask = F.interpolate(noise_mask.permute([0, 3, 1, 2]), size=(image.shape[1], image.shape[2]), mode='bicubic', align_corners=False)
            noise_mask = noise_mask.permute([0, 2, 3, 1])
        # Ensure we have the same number of masks and images
        if noise_mask.shape[0] > image.shape[0]:
            noise_mask = noise_mask[:image.shape[0]]
        else:
            noise_mask = torch.cat((noise_mask, noise_mask[-1:].repeat((image.shape[0]-noise_mask.shape[0], 1, 1, 1))), dim=0)

        # Convert mask to grayscale mask
        noise_mask = noise_mask.mean(dim=3).unsqueeze(-1)

        # add color noise
        imgs = image.clone().permute([0, 3, 1, 2])
        if color_noise > 0:
            color_noise = torch.normal(torch.zeros_like(imgs), std=color_noise)
            color_noise *= (imgs - imgs.min()) / (imgs.max() - imgs.min())

            imgs = imgs + color_noise
            imgs = imgs.clamp(0, 1)

        # create fine and coarse noise
        fine_noise = []
        for n in imgs:
            avg_color = n.mean(dim=[1,2])

            tmp_noise = T.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, fill=avg_color.tolist())(n)
            if blur > 0:
                tmp_noise = T.GaussianBlur(blur_size, blur)(tmp_noise)
            tmp_noise = T.ColorJitter(contrast=(contrast,contrast), saturation=(saturation,saturation))(tmp_noise)
            fine_noise.append(tmp_noise)

        imgs = None
        del imgs

        fine_noise = torch.stack(fine_noise, dim=0)
        fine_noise = fine_noise.permute([0, 2, 3, 1])
        #fine_noise = torch.stack(fine_noise, dim=0)
        #fine_noise = pb(fine_noise)
        mask_scale_diff = min(mask_scale_diff, 0.99)
        if mask_scale_diff > 0:
            coarse_noise = F.interpolate(fine_noise.permute([0, 3, 1, 2]), scale_factor=1-mask_scale_diff, mode='area')
            coarse_noise = F.interpolate(coarse_noise, size=(fine_noise.shape[1], fine_noise.shape[2]), mode='bilinear', align_corners=False)
            coarse_noise = coarse_noise.permute([0, 2, 3, 1])
        else:
            coarse_noise = fine_noise

        output = (1 - noise_mask) * coarse_noise + noise_mask * fine_noise

        if mask_strength < 1:
            noise_mask = noise_mask.pow(mask_strength)
            noise_mask = torch.nan_to_num(noise_mask).clamp(0, 1)
        output = noise_mask * output + (1 - noise_mask) * image

        # apply noise to image
        output = output * noise_strenght + image * (1 - noise_strenght)
        output = output.clamp(0, 1)

        return (output, )

IMAGE_CLASS_MAPPINGS = {
    # Image analysis
    "ImageEnhanceDifference+": ImageEnhanceDifference,

    # Image batch
    "ImageBatchMultiple+": ImageBatchMultiple,
    "ImageExpandBatch+": ImageExpandBatch,
    "ImageFromBatch+": ImageFromBatch,
    "ImageListToBatch+": ImageListToBatch,
    "ImageBatchToList+": ImageBatchToList,

    # Image manipulation
    "ImageCompositeFromMaskBatch+": ImageCompositeFromMaskBatch,
    "ImageComposite+": ImageComposite,
    "ImageCrop+": ImageCrop,
    "ImageFlip+": ImageFlip,
    "ImageRandomTransform+": ImageRandomTransform,
    "ImageRemoveAlpha+": ImageRemoveAlpha,
    "ImageRemoveBackground+": ImageRemoveBackground,
    "ImageResize+": ImageResize,
    "ImageSeamCarving+": ImageSeamCarving,
    "ImageTile+": ImageTile,
    "ImageUntile+": ImageUntile,
    "RemBGSession+": RemBGSession,
    "TransparentBGSession+": TransparentBGSession,

    # Image processing
    "ImageApplyLUT+": ImageApplyLUT,
    "ImageCASharpening+": ImageCAS,
    "ImageDesaturate+": ImageDesaturate,
    "PixelOEPixelize+": PixelOEPixelize,
    "ImagePosterize+": ImagePosterize,
    "ImageColorMatch+": ImageColorMatch,
    "ImageColorMatchAdobe+": ImageColorMatchAdobe,
    "ImageHistogramMatch+": ImageHistogramMatch,
    "ImageSmartSharpen+": ImageSmartSharpen,

    # Utilities
    "GetImageSize+": GetImageSize,
    "ImageToDevice+": ImageToDevice,
    "ImagePreviewFromLatent+": ImagePreviewFromLatent,
    "NoiseFromImage+": NoiseFromImage,
    #"ExtractKeyframes+": ExtractKeyframes,
}

IMAGE_NAME_MAPPINGS = {
    # Image analysis
    "ImageEnhanceDifference+": "🔧 Image Enhance Difference",

    # Image batch
    "ImageBatchMultiple+": "🔧 Images Batch Multiple",
    "ImageExpandBatch+": "🔧 Image Expand Batch",
    "ImageFromBatch+": "🔧 Image From Batch",
    "ImageListToBatch+": "🔧 Image List To Batch",
    "ImageBatchToList+": "🔧 Image Batch To List",

    # Image manipulation
    "ImageCompositeFromMaskBatch+": "🔧 Image Composite From Mask Batch",
    "ImageComposite+": "🔧 Image Composite",
    "ImageCrop+": "🔧 Image Crop",
    "ImageFlip+": "🔧 Image Flip",
    "ImageRandomTransform+": "🔧 Image Random Transform",
    "ImageRemoveAlpha+": "🔧 Image Remove Alpha",
    "ImageRemoveBackground+": "🔧 Image Remove Background",
    "ImageResize+": "🔧 Image Resize",
    "ImageSeamCarving+": "🔧 Image Seam Carving",
    "ImageTile+": "🔧 Image Tile",
    "ImageUntile+": "🔧 Image Untile",
    "RemBGSession+": "🔧 RemBG Session",
    "TransparentBGSession+": "🔧 InSPyReNet TransparentBG",

    # Image processing
    "ImageApplyLUT+": "🔧 Image Apply LUT",
    "ImageCASharpening+": "🔧 Image Contrast Adaptive Sharpening",
    "ImageDesaturate+": "🔧 Image Desaturate",
    "PixelOEPixelize+": "🔧 Pixelize",
    "ImagePosterize+": "🔧 Image Posterize",
    "ImageColorMatch+": "🔧 Image Color Match",
    "ImageColorMatchAdobe+": "🔧 Image Color Match Adobe",
    "ImageHistogramMatch+": "🔧 Image Histogram Match",
    "ImageSmartSharpen+": "🔧 Image Smart Sharpen",

    # Utilities
    "GetImageSize+": "🔧 Get Image Size",
    "ImageToDevice+": "🔧 Image To Device",
    "ImagePreviewFromLatent+": "🔧 Image Preview From Latent",
    "NoiseFromImage+": "🔧 Noise From Image",
}
