#!/usr/bin/env python
"""The run script."""
import logging
import os
import sys

# import flywheel functions
from flywheel_gear_toolkit import GearToolkitContext
from utils.parser import parse_config

from options.test_options import TestOptions
from models import create_model
import SimpleITK as sitk
from app.main import inference
from app.main import Registration

# import app.main as main


# Add top-level package directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Verify sys.path
print("sys.path:", sys.path)

# The gear is split up into 2 main components. The run.py file which is executed
# when the container runs. The run.py file then imports the rest of the gear as a
# module.

log = logging.getLogger(__name__)

def main(context: GearToolkitContext) -> None:
    # """Parses config and runs."""
    # subject_label, session_label, input_label = parse_config(context)
    
    print('Parsing config')
    opt = TestOptions().parse()

    image = sitk.ReadImage(opt.image)
    reference = sitk.ReadImage(opt.reference)
    workDir = '/flywheel/v0/work/tmp'

    print('Registering images')
    input_image, reference = Registration(image, reference, workDir)
    # sitk.WriteImage(image, outPath)

    print('Creating model')
    model = create_model(opt)
    model.setup(opt)

    print('Running inference')
    inference(model, input_image, opt.result_sr, opt.resample, opt.new_resolution, opt.patch_size[0],
              opt.patch_size[1], opt.patch_size[2], opt.stride_inplane, opt.stride_layer, 1)

    # have and option if gpu is called then have opt.result_gambas instead of opt.result_srs

# Only execute if file is run as main, not when imported by another module
if __name__ == "__main__":  # pragma: no cover
    # Get access to gear config, inputs, and sdk client if enabled.
    with GearToolkitContext() as gear_context:

        # Initialize logging, set logging level based on `debug` configuration
        # key in gear config.
        gear_context.init_logging()

        # Pass the gear context into main function defined above.
        main(gear_context)
