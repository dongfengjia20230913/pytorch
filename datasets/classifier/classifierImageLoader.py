import _init_paths

import os
import sys
from classifier.classifierImage import ClassfierImage

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from PIL import Image

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    return filename.lower().endswith(extensions)

def is_valid_file(x: str) -> bool:
    return has_file_allowed_extension(x, IMG_EXTENSIONS)

def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],) -> List[Tuple[str, int]]:

    instances = []#struct

    if not os.path.isdir(directory):
        raise ValueError("Image not dir!!!")

    image_count = 0
    for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        #print(item)
                        instances.append(item)

    return instances

#根据传入的目录名称，加载对应的分类名称和id，名称会按照升序排序
def find_classes(dir:str) -> Tuple[List[str], Dict[str, int]]:
    classes = [d.name for d in os.scandir(dir) if d.is_dir]
    classes.sort()
    class_to_idx = {cls_name:i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)