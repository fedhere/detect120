# detect120


**monitor the 120 Hz grid frequency with low cost equipment**

This repository collects the pipeline and some output from our experiments on studying the urban electrical grid's behavior through visible imaging monitoring at subsecond frequency. 
A single camera sited at an urban vantage point offers broad, persistent, real-time, and non-permissive coverage granular to individual housing units.

The 60 Hz AC line frequency of an electrical grid in the US induces a 120 Hz flicker in most of lights that it powers, including incandescent, halogen, transitional fluorescent, and some LED sources.1 This flicker is generally imperceptible to the unaided eye.
The 60 Hz line frequency is universal across the grid and maintained to within ∼ 0.02 Hz but the phase of the volt- age driving any particular light (and hence of its flicker) will depend upon the grid’s generating sources, topology and condition of its reactive components (e.g., distribution transformers), and the load local to the light. 
Changes in the phase, as observed in the urban lightscape, indicate changes in the power load. 

To avoid the use of expensive equipment necessary to monitor the 120Hz phase through high speed imaging we chop the image at near-line frequency (119.75 Hz) with a liquid-crystal shutter and thus down-convert the flicker to a beat frequency of ∼ 0.25 Hz, which is then easily imaged at 4 Hz cadence with a small digital camera.

This repository collects the data analysis pipeline for this project, from source selection, to phase identification, and the code used to generate the plots in our publications, plus additional material (many many additioinal figures, simulations, tests). The data is however not made public, since persistent imaging of urban landscapes is a practice susceptible to privacy concerns.

