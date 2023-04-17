"""test
Usage:
    test.py [--bulk_data] [--HE_image] [--sc_ref] [--spot_data]
    test.py (-h | --help)
    test.py --version

Options:
    -h --help   Show this screen.
    -v --version    Show version.
"""


import pandas as pd
import numpy as np
from docopt import docopt
from CytoBulk import *


def main(arguments):

    bulk_exp = arguments.get("--bulk_data")
    image_data = arguments.get("--HE_image")
    ref_sc = arguments.get("--sc_ref")
    spot_data = arguments.get("--spot_data")
    




if __name__=="__main__":
    arguments = docopt(__doc__, version="test 1.0.0")
    main(arguments)