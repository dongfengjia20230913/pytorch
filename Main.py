import _init_paths

import os
import sys
from classifier.classifierImage import ClassfierImage

from classifier.smokeDataset import SmokeData

from typing import Any, Callable, cast, Dict, List, Optional, Tuple




train_dir = './datas/smoke_clas/train'
SmokeData(train_dir)