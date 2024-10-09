# Thesis

Repository for all the experimental work for the thesis project

This repository will be temporary, the final output will be pushed to another repository once everything has been finalized

### Using SynthStrip Docker for skull-stripping MRI scans

1. Install Docker
2. Run the following command in your terminal to download SynthStrip:
   `docker pull freesurfer/synthstrip`
3. Verify the image in the terminal:
   `docker images`
4. Add Docker to your PATH environment variable
5. Navigate to `notebooks/` and open the file `skull_stripping_process.ipynb`
6. Update the `DOCKER_PATH` according to the location of your `docker.exe`
7. Ensure that `SOURCE_DIR` in the notebook points to the correct location of your VALDO Dataset
8. Run the Jupyter notebook
