


<p align="center"> <a href="https://mapir.isa.uma.es/mapirwebsite/?p=1792">Jose-Luis Matez-Bandera</a><sup>1</sup>, <a href="https://mapir.isa.uma.es/mapirwebsite/?p=1438">Javier Monroy</a><sup>1</sup> and <a href="http://mapir.isa.uma.es/jgonzalez">Javier Gonzalez-Jimenez</a><sup>1</sup> </p>

<p align="center"> <sup>1</sup> Machine Perception and Intelligent Robotics (MAPIR) Group,<br/> Malaga Institute for Mechatronics Engineering and Cyber-Physical Systems (IMECH.UMA).<br/> University of Malaga. Spain. </p>

### Content
<p align="center"> <a href="#abstract">Abstract</a>&nbsp;&nbsp;&nbsp;<a href="#demonstrationvideo">Demonstration Video</a>&nbsp;&nbsp;&nbsp;<a href="#code">Code</a>&nbsp;&nbsp;&nbsp;<a href="#methodoverview">Method Overview</a>&nbsp;&nbsp;&nbsp;<a href="#qualitativeresults">Qualitative Results</a>&nbsp;&nbsp;&nbsp;<a href="#citation">Citation</a></p>

### Abstract
This work presents a novel 3D reconstruction method to obtain the floor plan of a multi-room environment from a sequence of RGB-D images captured by a wheeled mobile robot. For each input image, the planar patches of visible walls are extracted and subsequently characterized by a multivariate Gaussian distribution in the convenient Plane Parameter Space. Then, accounting for the probabilistic nature of the robot localization, we transform and combine the planar patches from the camera frame into a 3D global model, where the planar patches include both the plane estimation uncertainty and the propagation of the robot pose uncertainty. Additionally, processing depth data, we detect openings (doors and windows) in the wall, which are also incorporated in the 3D global model to provide a more realistic representation. Experimental results, in both real-world and synthetic environments, demonstrate that our method outperforms state-of-the-art methods, both in time and accuracy, while just relying on Atlanta world assumption.


### Demonstration&nbsp;Video

[![Watch the video](https://img.youtube.com/vi/cFv2LAA0vMg/maxresdefault.jpg)](https://youtu.be/cFv2LAA0vMg)

### Code

### Method&nbsp;Overview

### Qualitative&nbsp;Results

### Citation
