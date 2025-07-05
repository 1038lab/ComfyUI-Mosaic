import torch
import cv2
import numpy as np

def create_pixelation_mosaic(image_tensor, block_size=8, mask=None):
    image_np = image_tensor.cpu().numpy()
    h, w, c = image_np.shape
    small_h, small_w = max(1, h // block_size), max(1, w // block_size)
    small = cv2.resize(image_np, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    result_tensor = torch.from_numpy(pixelated).float()
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, c)
        result_tensor = result_tensor * mask_expanded + image_tensor * (1.0 - mask_expanded)
    return result_tensor

def create_blur_mosaic(image_tensor, block_size=8, mask=None):
    image_np = image_tensor.cpu().numpy()
    h, w, c = image_np.shape
    kernel_size = max(3, block_size * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
    result_tensor = torch.from_numpy(blurred).float()
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, c)
        result_tensor = result_tensor * mask_expanded + image_tensor * (1.0 - mask_expanded)
    return result_tensor

def create_block_mosaic(image_tensor, block_size=8, mask=None):
    """Block average with soft edges - smoother transition between blocks"""
    image_np = image_tensor.cpu().numpy()
    h, w, c = image_np.shape
    result = image_np.copy()

    # Create overlapping blocks for smoother effect
    overlap = block_size // 4
    for y in range(0, h, block_size - overlap):
        for x in range(0, w, block_size - overlap):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            block = image_np[y:y_end, x:x_end]

            if block.size > 0:
                avg_color = np.mean(block, axis=(0, 1))

                # Create soft transition weights
                block_h, block_w = block.shape[:2]
                for by in range(block_h):
                    for bx in range(block_w):
                        # Distance from center
                        center_y, center_x = block_h // 2, block_w // 2
                        dist_y = abs(by - center_y) / (block_h / 2)
                        dist_x = abs(bx - center_x) / (block_w / 2)
                        weight = max(0, 1 - max(dist_y, dist_x))

                        if y + by < h and x + bx < w:
                            original = result[y + by, x + bx]
                            result[y + by, x + bx] = original * (1 - weight) + avg_color * weight

    result_tensor = torch.from_numpy(result).float()
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, c)
        result_tensor = result_tensor * mask_expanded + image_tensor * (1.0 - mask_expanded)
    return result_tensor

def create_custom_pattern_mosaic(image_tensor, block_size=8, pattern="squares", mask=None):
    image_np = image_tensor.cpu().numpy()
    h, w, c = image_np.shape
    result = image_np.copy()

    if pattern == "circles":
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = image_np[y:y+block_size, x:x+block_size]
                if block.size > 0:
                    avg_color = np.mean(block, axis=(0, 1))
                    center_y, center_x = block_size // 2, block_size // 2
                    radius = block_size // 3
                    for by in range(block_size):
                        for bx in range(block_size):
                            if (by - center_y)**2 + (bx - center_x)**2 <= radius**2:
                                if y + by < h and x + bx < w:
                                    result[y + by, x + bx] = avg_color

    elif pattern == "hexagons":
        hex_size = block_size // 2
        for y in range(0, h, int(hex_size * 1.5)):
            for x in range(0, w, int(hex_size * np.sqrt(3))):
                offset_x = hex_size if (y // int(hex_size * 1.5)) % 2 else 0
                center_x = x + offset_x
                center_y = y

                if center_y < h and center_x < w:
                    block_y1 = max(0, center_y - hex_size)
                    block_y2 = min(h, center_y + hex_size)
                    block_x1 = max(0, center_x - hex_size)
                    block_x2 = min(w, center_x + hex_size)

                    block = image_np[block_y1:block_y2, block_x1:block_x2]
                    if block.size > 0:
                        avg_color = np.mean(block, axis=(0, 1))

                        for by in range(block_y1, block_y2):
                            for bx in range(block_x1, block_x2):
                                dx = abs(bx - center_x)
                                dy = abs(by - center_y)
                                if dx <= hex_size and dy <= hex_size * 0.866:
                                    result[by, bx] = avg_color

    else:  # squares - sharp block boundaries
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = image_np[y:y+block_size, x:x+block_size]
                if block.size > 0:
                    avg_color = np.mean(block, axis=(0, 1))
                    result[y:y+block_size, x:x+block_size] = avg_color

    result_tensor = torch.from_numpy(result).float()
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, c)
        result_tensor = result_tensor * mask_expanded + image_tensor * (1.0 - mask_expanded)
    return result_tensor

def create_gradient_mosaic(image_tensor, block_size=8, direction="horizontal", mask=None):
    image_np = image_tensor.cpu().numpy()
    h, w, c = image_np.shape
    result = image_np.copy()

    if direction == "horizontal":
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = image_np[y:y+block_size, x:x+block_size]
                if block.size > 0:
                    left_color = np.mean(block[:, :block_size//2], axis=(0, 1))
                    right_color = np.mean(block[:, block_size//2:], axis=(0, 1))

                    for bx in range(block_size):
                        if x + bx < w:
                            alpha = bx / block_size
                            gradient_color = left_color * (1 - alpha) + right_color * alpha
                            for by in range(block_size):
                                if y + by < h:
                                    result[y + by, x + bx] = gradient_color

    elif direction == "vertical":
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = image_np[y:y+block_size, x:x+block_size]
                if block.size > 0:
                    top_color = np.mean(block[:block_size//2, :], axis=(0, 1))
                    bottom_color = np.mean(block[block_size//2:, :], axis=(0, 1))

                    for by in range(block_size):
                        if y + by < h:
                            alpha = by / block_size
                            gradient_color = top_color * (1 - alpha) + bottom_color * alpha
                            for bx in range(block_size):
                                if x + bx < w:
                                    result[y + by, x + bx] = gradient_color

    else:
        return create_blur_mosaic(image_tensor, block_size, mask)

    result_tensor = torch.from_numpy(result).float()
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, c)
        result_tensor = result_tensor * mask_expanded + image_tensor * (1.0 - mask_expanded)
    return result_tensor

def adjust_mosaic_intensity(original, processed, intensity):
    return original * (1.0 - intensity) + processed * intensity


class MosaicCreator:
    CATEGORY = "🧪AILab/🔦Mosaic"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "processing_mask")
    FUNCTION = "create_mosaic"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",),
                "mosaic_type": ([
                    "pixelation", 
                    "blur", 
                    "block_average", 
                    "squares", 
                    "circles", 
                    "hexagons",
                    "gradient_horizontal",
                    "gradient_vertical"
                ],),
                "block_size": ("INT", {
                    "default": 20, 
                    "min": 2, 
                    "max": 100,
                    "step": 1
                }),
                "intensity": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "preserve_edges": ("BOOLEAN", {"default": False}),
            }
        }
    
    def create_mosaic(self, image, mosaic_type, block_size, intensity, mask=None, preserve_edges=False):
        try:
            # ComfyUI image format: [B, H, W, C]
            # ComfyUI mask format: [B, H, W]

            # Ensure block_size is integer
            block_size = int(block_size)
            intensity = float(intensity)

            batch_size = image.shape[0]
            results = []

            for i in range(batch_size):
                current_image = image[i]  # [H, W, C]
                current_mask = mask[i] if mask is not None else None  # [H, W]

                # Create mosaic effect
                if mosaic_type == "pixelation":
                    mosaic_result = create_pixelation_mosaic(current_image, block_size, current_mask)
                elif mosaic_type == "blur":
                    mosaic_result = create_blur_mosaic(current_image, block_size, current_mask)
                elif mosaic_type == "block_average":
                    mosaic_result = create_block_mosaic(current_image, block_size, current_mask)
                elif mosaic_type in ["squares", "circles", "hexagons"]:
                    mosaic_result = create_custom_pattern_mosaic(
                        current_image, block_size, mosaic_type, current_mask
                    )
                elif mosaic_type == "gradient_horizontal":
                    mosaic_result = create_gradient_mosaic(
                        current_image, block_size, "horizontal", current_mask
                    )
                elif mosaic_type == "gradient_vertical":
                    mosaic_result = create_gradient_mosaic(
                        current_image, block_size, "vertical", current_mask
                    )
                else:
                    mosaic_result = create_pixelation_mosaic(current_image, block_size, current_mask)

                # Apply intensity adjustment
                if intensity < 1.0:
                    mosaic_result = adjust_mosaic_intensity(current_image, mosaic_result, intensity)

                results.append(mosaic_result)

            # Combine results back to batch format
            final_image = torch.stack(results, dim=0)

            # Return the input mask as output mask (or create full mask if none)
            if mask is not None:
                output_mask = mask
            else:
                output_mask = torch.ones((batch_size, image.shape[1], image.shape[2]), dtype=torch.float32)

            return (final_image, output_mask)

        except Exception as e:
            print(f"Error in MosaicCreator: {str(e)}")
            import traceback
            traceback.print_exc()

            # Return original image on error
            if mask is not None:
                return (image, mask)
            else:
                error_mask = torch.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)
                return (image, error_mask)


class MosaicDetector:
    CATEGORY = "🧪AILab/🔦Mosaic"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask_overlay", "mask")
    FUNCTION = "detect_mosaic"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "top_n": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 2}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "overlay_color": (["red", "green", "blue", "yellow", "cyan", "magenta", "white"],),
                "overlay_opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    def detect_mosaic(self, image, top_n, mask_expand, mask_blur, invert_mask, overlay_color, overlay_opacity):
        try:
            batch_size = image.shape[0]
            result_images = []
            result_overlays = []
            result_masks = []

            color_map = {
                "red": [1.0, 0.0, 0.0],
                "green": [0.0, 1.0, 0.0],
                "blue": [0.0, 0.0, 1.0],
                "yellow": [1.0, 1.0, 0.0],
                "cyan": [0.0, 1.0, 1.0],
                "magenta": [1.0, 0.0, 1.0],
                "white": [1.0, 1.0, 1.0]
            }

            for i in range(batch_size):
                current_image = image[i]
                image_np = (current_image.cpu().numpy() * 255).astype(np.uint8)

                img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                img_gray = cv2.Canny(img_gray, 10, 20)
                img_gray = 255 - img_gray
                img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)

                mask = np.zeros_like(img_gray, dtype=np.uint8)

                for size in range(10, 21):
                    template = self.create_grid_pattern(size)
                    w, h = template.shape[::-1]
                    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.3
                    loc = np.where(result >= threshold)
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(mask, pt, (pt[0] + w, pt[1] + h), 255, -1)

                mask = self.keep_largest_components(mask, top_n)

                if mask_expand > 0:
                    kernel = np.ones((mask_expand, mask_expand), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)

                if mask_blur > 0:
                    blur_size = mask_blur if mask_blur % 2 == 1 else mask_blur + 1
                    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

                mask_tensor = torch.from_numpy(mask).float() / 255.0

                if invert_mask:
                    mask_tensor = 1.0 - mask_tensor

                overlay_rgb = color_map[overlay_color]
                overlay_image = current_image.clone()
                for c in range(3):
                    overlay_image[:, :, c] = (
                        current_image[:, :, c] * (1.0 - mask_tensor * overlay_opacity) +
                        overlay_rgb[c] * mask_tensor * overlay_opacity
                    )

                result_images.append(current_image)
                result_overlays.append(overlay_image)
                result_masks.append(mask_tensor)

            final_images = torch.stack(result_images, dim=0)
            final_overlays = torch.stack(result_overlays, dim=0)
            final_masks = torch.stack(result_masks, dim=0)

            return (final_images, final_overlays, final_masks)

        except Exception as e:
            print(f"Error in MosaicDetector: {str(e)}")
            error_mask = torch.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype=torch.float32)
            return (image, image, error_mask)

    def create_grid_pattern(self, size):
        pattern_size = size + 4
        pattern = np.ones((pattern_size, pattern_size), dtype=np.uint8) * 255

        for i in range(2, pattern_size, size - 1):
            for j in range(pattern_size):
                pattern[i, j] = 0
                pattern[j, i] = 0

        return pattern

    def keep_largest_components(self, mask, top_n):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask

        areas = stats[1:, cv2.CC_STAT_AREA]
        top_indices = np.argsort(areas)[-top_n:]

        result_mask = np.zeros_like(mask)
        for idx in top_indices:
            result_mask[labels == (idx + 1)] = 255

        return result_mask
