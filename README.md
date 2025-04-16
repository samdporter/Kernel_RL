# Simple Repository demonstrating Kernelised Richardson Lucy Method

Comparisons with simple RL and MAPRL with directional total variation

## To run...

First run `python3 dependencies.py` to check whetehr the required packages are installed. This repository requires SIRF to be installed, which cannot be done thorugh pip and conda. 

A lot of the code should be easily translatable to CIL `DataContainers` as SIRF projectors are only needed for the creation of simulated OSEM images.

Then run `python3 setup_demo.py` to create OSEM images and a point spread functions (used to estimate PSF - not a particularly good method)

Finally run `python3 run_deconv.py` to compare the three methods. This will create plots of objective functions, RMSEs and compare output images. RMSE is currently not performed for the MAPRL because it would require a CIL `Callback` method that is currently not available on the CS clsuter @ UCL.