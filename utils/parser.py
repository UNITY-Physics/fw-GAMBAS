"""Parser module to parse gear config.json."""

from typing import Tuple
import os
import re
from flywheel_gear_toolkit import GearToolkitContext

import os
import subprocess

def check_gpu():
    """Check if the container has access to a GPU."""
    try:
        # Check if NVIDIA GPUs are available
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("GPU detected!")
            return True
        else:
            print("No GPU detected.")
            return False
    except FileNotFoundError:
        print("nvidia-smi not found. No GPU available.")
        return False


def parse_config(context):
    """Parse the config and other options from the context, both gear and app options.

    Returns:
        gear_inputs
        gear_options: options for the gear
        app_options: options to pass to the app
    """

    is_gpu = check_gpu()
    if is_gpu:
        print("Running on GPU")
        model = 'GAMBAS'
    else:
        print("Running on CPU")
        model = 'ResCNN'
    
    # Get the input file id
    input_container = context.client.get_analysis(context.destination["id"])

    # Get the subject id from the session id
    # & extract the subject container
    subject_id = input_container.parents['subject']
    subject_container = context.client.get(subject_id)
    subject = subject_container.reload()
    print("subject label: ", subject.label)
    subject_label = subject.label

    # Get the session id from the input file id
    # & extract the session container
    session_id = input_container.parents['session']
    session_container = context.client.get(session_id)
    session = session_container.reload()
    session_label = session.label
    session_label = session_label.replace(" ", "_")
    print("session label: ", session.label)

    # Specify the directory you want to list files from
    directory_path = '/flywheel/v0/input/input'

    # Define relevant keywords to keep
    allowed_keywords = ["T2w", "T1w", "T2", "T1", "AXI", "SAG", "COR", "Fast"]

    # Define diagnostic-related terms to remove
    diagnostic_terms = ["DIAGNOSTIC", "NOT_FOR_DIAGNOSTIC_USE", "NOTDIAGNOSTIC"]

    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            filename_without_extension = filename.rsplit('.', 1)[0]  # Remove file extension
            
            # Remove diagnostic-related terms (case insensitive)
            for term in diagnostic_terms:
                filename_without_extension = re.sub(term, '', filename_without_extension, flags=re.IGNORECASE)

            # Replace non-alphanumeric characters with underscores (preserve letters/numbers)
            cleaned_string = re.sub(r'[^a-zA-Z0-9]', '_', filename_without_extension)
            
            # Remove trailing underscores
            cleaned_string = cleaned_string.rstrip('_')

            # Extract relevant keywords
            extracted_keywords = [word for word in allowed_keywords if word in cleaned_string]

            # Ensure "T1" and "T2" are converted to "T1w" and "T2w"
            formatted_keywords = [
                "T1w" if word == "T1" else "T2w" if word == "T2" else word
                for word in extracted_keywords
            ]

            # Construct the final input label
            if formatted_keywords:
                input_label = "_".join(formatted_keywords)
            else:
                # Fallback: remove leading numbers and underscores
                input_label = re.sub(r'^\d+_?', '', cleaned_string)

            print("Input label:", input_label)

            output_label = f'sub-{subject_label}_ses-{session_label}_acq-{input_label}_rec-{model}.nii.gz'
            print("Output filename:", output_label)
    
    return output_label, model