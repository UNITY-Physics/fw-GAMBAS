# Changelog

11/03/2025
Version 0.1.5 NJB
Options for running on GPU or CPU should be available (requires testing)

26/02/2025

Version 0.0.5 NJB
- Successfully tested on Flywheel

To Do:
clean output filename

Version 0.0.2 NJB
- refactoring the parser code

To Do:
** base_options needs to be updated to handle gpu selection rather than cpu default
** need to rebuild base image to handle GPU selection
** To run on GPU should batch process to save costs of loading Docker container each time


20/02/2025
Version 0.0.2 NJB, LB
- updates to run on CPU
- Not pushed to FW, need to rebuild from CUDA Docker image to ensure compatibility to run on GPU
- Current default is to run on CPU, need to check option to run on GPU
    This is currently set as condition if CUDA is available, then run on GPU, else run on CPU

14/02/2025
Version 0.0.1

Refactoring for Flywheel gear