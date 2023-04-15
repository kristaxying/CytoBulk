"""ImmuneGraph
Usage:
    ImmuneGraph.py [--bulk_data] [--HE_image] [--sc_ref] [--spot_data] [--species=<sn>]
    ImmuneGraph.py (-h | --help)
    ImmuneGraph.py --version

Options:
    -h --help   Show this screen.
    -v --version    Show version.
"""


import pandas as pd
import numpy as np
from docopt import docopt
import datetime
import CytoBulk


def main(arguments):
    #st_data = arguments.get("<st_data>")
    #st_meta = arguments.get("<st_meta>")
    #h5_file = arguments.get("<h5_file>")
    st_files = arguments.get("<st_file>")
    lr_pair = arguments.get("<lr_pair>")
    pathwaydb = arguments.get("<pathwaydb>")
    cell_sender = str(arguments.get("--cell_sender"))
    cell_receiver = str(arguments.get("--cell_receiver"))
    parallel = arguments.get("--parallel")
    filtering = str(arguments.get("--filtering"))
    species = str(arguments.get("--species"))
    distance_threshold = arguments.get("--distance_threshold")
    out_dir = str(arguments.get("--out"))
    max_hop = int(arguments.get("--max_hop"))
    n_core = int(arguments.get("--core"))
    starttime = datetime.datetime.now()




if __name__=="__main__":
    arguments = docopt(__doc__, version="CytoTour 1.0.0")
    main(arguments)