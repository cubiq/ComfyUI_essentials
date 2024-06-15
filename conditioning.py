from nodes import MAX_RESOLUTION, ConditioningZeroOut, ConditioningSetTimestepRange, ConditioningCombine

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
    CATEGORY = "essentials/conditioning"

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
    CATEGORY = "essentials/conditioning"

    def execute(self, conditioning_1, conditioning_2, conditioning_3=None, conditioning_4=None, conditioning_5=None):
        c = conditioning_1 + conditioning_2

        if conditioning_3 is not None:
            c += conditioning_3
        if conditioning_4 is not None:
            c += conditioning_4
        if conditioning_5 is not None:
            c += conditioning_5
        
        return (c,)

class SD3NegativeConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "end": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001 }),
        }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "essentials/conditioning"

    def execute(self, conditioning, end):      
        zero_c = ConditioningZeroOut().zero_out(conditioning)[0]

        if end == 0:
            return (zero_c, )

        c = ConditioningSetTimestepRange().set_range(conditioning, 0, end)[0]
        zero_c = ConditioningSetTimestepRange().set_range(zero_c, end, 1.0)[0]      
        c = ConditioningCombine().combine(zero_c, c)[0]

        return (c, )

COND_CLASS_MAPPINGS = {
    "CLIPTextEncodeSDXL+": CLIPTextEncodeSDXLSimplified,
    "ConditioningCombineMultiple+": ConditioningCombineMultiple,
    "SD3NegativeConditioning+": SD3NegativeConditioning,
}

COND_NAME_MAPPINGS = {
    "CLIPTextEncodeSDXL+": "🔧 SDXL CLIPTextEncode",
    "ConditioningCombineMultiple+": "🔧 Cond Combine Multiple",
    "SD3NegativeConditioning+": "🔧 SD3 Negative Conditioning"
}