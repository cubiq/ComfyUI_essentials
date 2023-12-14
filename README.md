# :wrench: ComfyUI Essentials

Essential nodes that are weirdly missing from ComfyUI core. With few exceptions they are new features and not commodities. I hope this will be just a temporary repository until the nodes get included into ComfyUI.

## Node list

- Get Image Size
- Image Resize (adds "keep proportions" to scale image)
- Image Crop (includes auto crop to all sides and center)
- Image Flip
- Image Desaturate
- Image Posterize
- Image Contrast Adaptive Sharpening
- Image Enhance Difference
- Image Expand Batch, expands an image batch to a given size repeating the images uniformly
- Mask Blur
- Mask Flip
- ~~Mask Grow / Shrink (same as Mask grow but adds shrink)~~ (this was recently added in the official repo)
- Mask Preview
- Mask Batch, same as Image batch but for masks
- Mask Expand Batch, expands a mask batch to a given size repeating the masks uniformly
- Mask From Color
- Transition Mask, creates a transition with series of masks, useful for animations
- Simple Math
- Console Debug (outputs any input to console)
- Model Compile, will hurt your feelings. It basically compiles the model with torch.compile. It takes a few minutes to compile but generation is faster. Only works on Linux and Mac (maybe WLS I dunno)
- TODO: Mask Save
- TODO: documentation

Let me know if anything's missing
