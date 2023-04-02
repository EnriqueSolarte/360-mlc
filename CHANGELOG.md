# Changelog

## April 1st, 2023 by @EnriqueSolarte
* Added `mvl_data/load_and_eval_mvl_data.py` to evaluate mp3d_fpe data using HorizonNet (estimating boundaries), plot on the image, and plot the 3D boundary. 
* Fixed missed file `config/global_directories.yaml`
* Fixed wrong spherical projection. 
* Fixed issue of STD (mlc labels) zero or NAN. This arises with scenes with few frames (or highly noisy scenes), which result in a standard deviation close to zero. This issue can also occur when some points are missed at the boundary due to sparse point projection, resulting in a NAN loss when evaluating the standard deviation. To resolve this, we applied a blur kernel to the boundary reprojection map to ensure continuity along the wide 360-image.
