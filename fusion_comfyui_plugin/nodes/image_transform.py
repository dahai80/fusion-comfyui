# fusion_comfyui_plugin/nodes/image_transform.py
import logging

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.image_transform")


class ImageScale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                             "width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1}),
                             "height": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1}),
                             "crop": (s.crop_methods,)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, width, height, crop):
        if width == 0 and height == 0:
            logger.debug("ImageScale: passthrough %s", image.shape)
            return (image,)
        samples = np.transpose(image, (0, 3, 1, 2))
        if width == 0:
            width = max(1, round(samples.shape[3] * height / samples.shape[2]))
        elif height == 0:
            height = max(1, round(samples.shape[2] * width / samples.shape[3]))
        from nodes._scaling import common_upscale
        s = common_upscale(samples, width, height, upscale_method, crop)
        s = np.transpose(s, (0, 2, 3, 1))
        logger.info("ImageScale: %s -> %s", image.shape, s.shape)
        return (s,)


class ImageScaleBy:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                             "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, scale_by):
        samples = np.transpose(image, (0, 3, 1, 2))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        from nodes._scaling import common_upscale
        s = common_upscale(samples, width, height, upscale_method, "disabled")
        s = np.transpose(s, (0, 2, 3, 1))
        logger.info("ImageScaleBy: %s -> %s", image.shape, s.shape)
        return (s,)


class ImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image1": ("IMAGE",), "image2": ("IMAGE",)}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"
    CATEGORY = "image/batch"
    DEPRECATED = True

    def batch(self, image1, image2):
        if image1.shape[-1] != image2.shape[-1]:
            if image1.shape[-1] > image2.shape[-1]:
                image2 = np.pad(image2, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0)
            else:
                image1 = np.pad(image1, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1.0)
        if image1.shape[1:] != image2.shape[1:]:
            s2 = np.transpose(image2, (0, 3, 1, 2))
            from nodes._scaling import common_upscale
            s2 = common_upscale(s2, image1.shape[2], image1.shape[1], "bilinear", "center")
            image2 = np.transpose(s2, (0, 2, 3, 1))
        s = np.concatenate((image1, image2), axis=0)
        logger.info("ImageBatch: %s + %s -> %s", image1.shape, image2.shape, s.shape)
        return (s,)


class EmptyImage:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                             "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "image"

    def generate(self, width, height, batch_size=1, color=0):
        r = ((color >> 16) & 0xFF) / 0xFF
        g = ((color >> 8) & 0xFF) / 0xFF
        b = (color & 0xFF) / 0xFF
        img = np.full((batch_size, height, width, 3), [r, g, b], dtype=np.float32)
        logger.info("EmptyImage: %s color=#%06X", img.shape, color)
        return (img,)


class ImagePadForOutpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                             "top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                             "right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                             "bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                             "feathering": ("INT", {"default": 40, "min": 0, "max": 8192, "step": 1, "advanced": True}),}}
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"
    CATEGORY = "image/transform"

    def expand_image(self, image, left, top, right, bottom, feathering):
        d1, d2, d3, d4 = image.shape
        new_image = np.ones((d1, d2 + top + bottom, d3 + left + right, d4), dtype=np.float32) * 0.5
        new_image[:, top:top + d2, left:left + d3, :] = image
        mask = np.ones((d2 + top + bottom, d3 + left + right), dtype=np.float32)
        t = np.zeros((d2, d3), dtype=np.float32)
        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:
            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2
                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3
                    d = min(dt, db, dl, dr)
                    if d >= feathering:
                        continue
                    v = (feathering - d) / feathering
                    t[i, j] = v * v
        mask[top:top + d2, left:left + d3] = t
        logger.info("ImagePadForOutpaint: %s -> img %s mask %s", image.shape, new_image.shape, mask.shape)
        return (new_image, mask[np.newaxis, ...])


class LoadImageMask:
    _color_channels = ["alpha", "red", "green", "blue"]

    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        import os
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required": {"image": (sorted(files), {"image_upload": True}), "channel": (s._color_channels,)}}

    CATEGORY = "image"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "load_image_mask"

    def load_image_mask(self, image, channel):
        from nodes.image import LoadImage
        image_arr, mask_arr = LoadImage().load_image(image)
        # LoadImage (Task 7 not yet landed) may return torch tensors; np.asarray
        # produces a numpy view/copy for both torch and numpy inputs, keeping
        # this file torch-free regardless of upstream state.
        image_arr = np.asarray(image_arr)
        mask_arr = np.asarray(mask_arr)
        c = channel[0].upper()
        if c == "A":
            return (mask_arr,)
        channel_idx = {"R": 0, "G": 1, "B": 2}.get(c, 0)
        if channel_idx < image_arr.shape[-1]:
            return (np.ascontiguousarray(image_arr[..., channel_idx]).copy(),)
        empty = np.zeros(image_arr.shape[:-1], dtype=np.float32)
        return (empty,)

    @classmethod
    def IS_CHANGED(s, image, channel):
        from nodes.image import LoadImage
        return LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        import folder_paths
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
