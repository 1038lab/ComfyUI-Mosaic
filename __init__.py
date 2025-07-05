try:
    from .mosaic_creator import MosaicCreator, MosaicDetector
except ImportError:
    # Fallback for direct execution
    from mosaic_creator import MosaicCreator, MosaicDetector

NODE_CLASS_MAPPINGS = {
    "MosaicCreator": MosaicCreator,
    "MosaicDetector": MosaicDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MosaicCreator": "Mosaic Creator",
    "MosaicDetector": "Mosaic Detector",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
