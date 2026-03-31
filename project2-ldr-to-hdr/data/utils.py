import cv2
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS


def read_hdr(image_path: str) -> ndarray:
    """
    Read HDR image and convert it to the proper RGB layout.
    """
    hdr_image = cv2.imread(
        filename=image_path,
        flags=cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR
    )
    hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    return hdr_image

def get_exif(image_path: str) -> dict:
    """
    Read metadata from the image.
    Interesting values: "ExposureTime", "FNumber"
    """
    image = Image.open(image_path)
    info = image._getexif()
    
    if info is None:
      raise ValueError(f"Missing metadata for image {image_path}")
  
    exif_data = {}
    for tag, value in info.items():
        decoded = TAGS.get(tag)
        if not decoded:
            continue
        exif_data[decoded] = value

    return exif_data

def _calculate_luminance(hdr_image):
    return (0.2126 * hdr_image[:, :, 0] +
            0.7152 * hdr_image[:, :, 1] +
            0.0722 * hdr_image[:, :, 2])
    
def _filter_pixels(luminance, epsilon):
    return luminance[luminance > epsilon]

def measure_ev_range(hdr_image, epsilon=1e-6) -> float:
    """
    Measure dynamic range for HDR file.
    """
    valid_pixels = _filter_pixels(
        luminance=_calculate_luminance(hdr_image),
        epsilon=epsilon
    )
    
    luminance_max = np.max(valid_pixels)
    luminance_min = np.min(valid_pixels)
    
    return np.log2(luminance_max / luminance_min)

def tone_map_reinhard(hdr_image: ndarray) -> ndarray:
    """
    Tonemapping Reinhard's operator.
    """
    tonemap_operator = cv2.createTonemapReinhard(
        gamma=2.2,
        intensity=0.0,
        light_adapt=0.0,
        color_adapt=0.0
    )
    result = tonemap_operator.process(src=hdr_image)
    return result

HDR_PATH = "path_to_hdr"
LDR_PATH = "path_to_jpg"

if __name__ == "__main__":
    # read LDR image and show metadata
    metadata = get_exif(LDR_PATH)
    print(f"Exposure Time: {metadata.get('ExposureTime')}")
    print(f"FNumber: {metadata.get('FNumber')}")
    
    # read HDR image and calculate Dynamic Range
    hdr_img = read_hdr(HDR_PATH)
    print(f"Data type: {hdr_img.dtype}")
    print(f"Range: {hdr_img.min()} do {hdr_img.max()}")
    print(f"Dynamic Range: {measure_ev_range(hdr_img)}")
    
    # tonemap HDR image and show it on the screen
    gamma_corrected = tone_map_reinhard(hdr_img)
    plt.imshow(gamma_corrected)
    plt.show()