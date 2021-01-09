import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)


# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
net_path = osp.join(this_dir, 'net')
datgasets_path = osp.join(this_dir, 'datasets')

add_path(lib_path)
add_path(net_path)
add_path(datgasets_path)