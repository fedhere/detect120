# detect120


**monitor the 120 Hz grid frequency with low cost equipment**

This repository collects the pipeline and some output from our experiments on studying the urban electrical grid's behavior through visible imaging monitoring at subsecond frequency. 
A single camera sited at an urban vantage point offers broad, persistent, real-time, and non-permissive coverage granular to individual housing units.

The 60 Hz AC line frequency of an electrical grid in the US induces a 120 Hz flicker in most of lights that it powers, including incandescent, halogen, transitional fluorescent, and some LED sources.1 This flicker is generally imperceptible to the unaided eye.
The 60 Hz line frequency is universal across the grid and maintained to within ∼ 0.02 Hz but the phase of the volt- age driving any particular light (and hence of its flicker) will depend upon the grid’s generating sources, topology and condition of its reactive components (e.g., distribution transformers), and the load local to the light. 
Changes in the phase, as observed in the urban lightscape, indicate changes in the power load. 

To avoid the use of expensive equipment necessary to monitor the 120Hz phase through high speed imaging we chop the image at near-line frequency (119.75 Hz) with a liquid-crystal shutter and thus down-convert the flicker to a beat frequency of ∼ 0.25 Hz, which is then easily imaged at 4 Hz cadence with a small digital camera.

This repository collects the data analysis pipeline for this project, from source selection, to phase identification, and the code used to generate the plots in our publications, plus additional material (many many additioinal figures, simulations, tests). The data is however not made public, since persistent imaging of urban landscapes is a practice susceptible to privacy concerns.

The steps of the pipeline, and the codes that perform them,  are as follow.

1. First off: You need to decide what is a source. Generally that starts with stacking images to get a deep enought image of the night skyline. Use **stackImages.py**

for example:

>$python stackImages.py groundtest1/ESB_c0.7Hz_250ms_2016-05-24-230354 --nstack 20 --showme  --gif

This creates a directory **stacks** and stores the aa file recording the image size in it (under the assumptinon that the image size for science images is the same as that of the images used to make the stack.  If the image input has a path it will also create a directory corresponding to the full image path, up to the name (**groundtest1** in this case)

2. Find the windows in the stack image. We do that by high pass filtering the image and then threshold it. Use **windosFinder.py** The threshold is set automatically to 90% of the distribution of pixels. It can also be set by hand


3. Now you can extract the lightcurves and analyze them! the code that does all that is **getalllcvPCA.py**
This is a large piece of code (and the docstrings are on still on my todo list)

4. One should consider that some automatically selected windows belong to the same building, even to the same housing unit. Use **lassoselect.py** to create a file containing labels that identify sources grouped together. This is an interactive tool that selects groups of windows as you draw a lasso around them. 


4. 
