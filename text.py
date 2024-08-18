import os
import torch
from nodes import MAX_RESOLUTION
import torchvision.transforms.v2 as T
from .utils import FONTS_DIR

class DrawText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", { "multiline": True, "dynamicPrompts": True, "default": "Hello, World!" }),
                "font": (sorted([f for f in os.listdir(FONTS_DIR) if f.endswith('.ttf') or f.endswith('.otf')]), ),
                "size": ("INT", { "default": 56, "min": 1, "max": 9999, "step": 1 }),
                "color": ("STRING", { "multiline": False, "default": "#FFFFFF" }),
                "background_color": ("STRING", { "multiline": False, "default": "#00000000" }),
                "shadow_distance": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1 }),
                "shadow_blur": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1 }),
                "shadow_color": ("STRING", { "multiline": False, "default": "#000000" }),
                "horizontal_align": (["left", "center", "right"],),
                "vertical_align": (["top", "center", "bottom"],),
                "offset_x": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "offset_y": ("INT", { "default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1 }),
                "direction": (["ltr", "rtl"],),
            },
            "optional": {
                "img_composite": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/text"

    def execute(self, text, font, size, color, background_color, shadow_distance, shadow_blur, shadow_color, horizontal_align, vertical_align, offset_x, offset_y, direction, img_composite=None):
        from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter

        font = ImageFont.truetype(os.path.join(FONTS_DIR, font), size)

        lines = text.split("\n")
        if direction == "rtl":
            lines = [line[::-1] for line in lines]

        # Calculate the width and height of the text
        text_width = max(font.getbbox(line)[2] for line in lines)
        line_height = font.getmask(text).getbbox()[3] + font.getmetrics()[1]  # add descent to height
        text_height = line_height * len(lines)

        if img_composite is not None:
            img_composite = T.ToPILImage()(img_composite.permute([0,3,1,2])[0]).convert('RGBA')
            width = img_composite.width
            height = img_composite.height
            image = Image.new('RGBA', (width, height), color=background_color)
        else:
            width = text_width
            height = text_height
            background_color = ImageColor.getrgb(background_color)
            image = Image.new('RGBA', (width + shadow_distance, height + shadow_distance), color=background_color)

        image_shadow = None
        if shadow_distance > 0:
            image_shadow = image.copy()
            #image_shadow = Image.new('RGBA', (width + shadow_distance, height + shadow_distance), color=background_color)

        for i, line in enumerate(lines):
            line_width = font.getbbox(line)[2]
            #text_height =font.getbbox(line)[3]
            if horizontal_align == "left":
                x = 0
            elif horizontal_align == "center":
                x = (width - line_width) / 2
            elif horizontal_align == "right":
                x = width - line_width
            
            if vertical_align == "top":
                y = 0
            elif vertical_align == "center":
                y = (height - text_height) / 2
            elif vertical_align == "bottom":
                y = height - text_height

            x += offset_x
            y += i * line_height + offset_y

            draw = ImageDraw.Draw(image)
            draw.text((x, y), line, font=font, fill=color)

            if image_shadow is not None:
                draw = ImageDraw.Draw(image_shadow)
                draw.text((x + shadow_distance, y + shadow_distance), line, font=font, fill=shadow_color)

        if image_shadow is not None:
            image_shadow = image_shadow.filter(ImageFilter.GaussianBlur(shadow_blur))
            image = Image.alpha_composite(image_shadow, image)

        #image = T.ToTensor()(image).unsqueeze(0).permute([0,2,3,1])
        mask = T.ToTensor()(image).unsqueeze(0).permute([0,2,3,1])
        mask = mask[:, :, :, 3] if mask.shape[3] == 4 else torch.ones_like(mask[:, :, :, 0])

        if img_composite is not None:
            image = Image.alpha_composite(img_composite, image)
        
        image = T.ToTensor()(image).unsqueeze(0).permute([0,2,3,1])

        return (image[:, :, :, :3], mask,)

TEXT_CLASS_MAPPINGS = {
    "DrawText+": DrawText,
}

TEXT_NAME_MAPPINGS = {
    "DrawText+": "🔧 Draw Text",
}