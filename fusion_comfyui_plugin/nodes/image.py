import hashlib
import json
import logging
import os
import random

import numpy as np

logger = logging.getLogger("fusion_comfyui.nodes.image")

disable_metadata = False
logger.debug("image node metadata embedding: %s", "disabled" if disable_metadata else "enabled")


class LoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image):
        from PIL import Image, ImageOps, ImageSequence
        from fusion_comfyui.core.bridge import to_image_tensor
        import folder_paths

        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            rgb = i.convert("RGB")

            if len(output_images) == 0:
                w = rgb.size[0]
                h = rgb.size[1]

            if rgb.size[0] != w or rgb.size[1] != h:
                continue

            arr = np.array(rgb).astype(np.float32) / 255.0
            output_images.append(arr)

            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - mask
            else:
                mask = np.zeros((h, w), dtype=np.float32)
            output_masks.append(mask)

        output_image = np.stack(output_images, axis=0)
        output_mask = np.stack(output_masks, axis=0)
        # Core IMAGE/MASK consumers (BatchImagesNode torch.cat, RepeatImageBatch,
        # MaskToImage) require torch tensors; wrap numpy -> CPU torch tensor.
        image_t = to_image_tensor(output_image)
        import torch
        mask_t = torch.from_numpy(np.ascontiguousarray(output_mask)).float()

        logger.info("LoadImage: %s shape=%s", image, tuple(image_t.shape))
        return (image_t, mask_t)

    @classmethod
    def IS_CHANGED(cls, image):
        import folder_paths
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        import folder_paths
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


class SaveImage:
    def __init__(self):
        import folder_paths
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        import folder_paths

        filename_prefix += self.prefix_append
        h, w = images.shape[1], images.shape[2]
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(filename_prefix, self.output_dir, h, w)
        )
        results = list()
        for batch_number, image in enumerate(images):
            if isinstance(image, np.ndarray):
                i = 255.0 * image
            else:
                i = 255.0 * np.array(image)
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        logger.info("SaveImage: saved %d images to %s", len(results), full_output_folder)
        return {"ui": {"images": results}, "result": (images,)}


class PreviewImage(SaveImage):
    def __init__(self):
        import folder_paths
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"images": ("IMAGE",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
