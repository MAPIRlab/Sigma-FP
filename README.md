# <p align="center"> Sigma-FP: Robot Mapping of 3D Floor Plans with an RGB-D Camera under Uncertainty </p>

<p align="center"> <a href="https://mapir.isa.uma.es/mapirwebsite/?p=1792">Jose-Luis Matez-Bandera</a><sup>1</sup>, <a href="https://mapir.isa.uma.es/mapirwebsite/?p=1438">Javier Monroy</a><sup>1</sup> and <a href="http://mapir.isa.uma.es/jgonzalez">Javier Gonzalez-Jimenez</a><sup>1</sup> </p>

<p align="center"> <sup>1</sup> Machine Perception and Intelligent Robotics (MAPIR) Group,<br/> Malaga Institute for Mechatronics Engineering and Cyber-Physical Systems (IMECH.UMA).<br/> University of Malaga. Spain. </p>

### Content
<p align="center"> <a href="#citation">Citation</a>&nbsp;&nbsp;&nbsp;<a href="#requirements">Requirements</a>&nbsp;&nbsp;&nbsp;<a href="#configuration">Configuration</a>&nbsp;&nbsp;&nbsp;<a href="#howtorun">How to Run</a></p>

### Citation
<pre><code>@article{matez2022,  
    title={Sigma-FP: Robot Mapping of 3D Floor Plans with an RGB-D Camera under Uncertainty},  
    author={Matez-Bandera, Jose-Luis and Monroy, Javier and Gonzalez-Jimenez, Javier},  
    year={2022},  
    note={Under Review}  
    }
</code></pre>

### Installation and Requirements

Clone the repository in the /src directory of your ROS workspace:

<code>git clone https://github.com/MAPIRlab/Sigma-FP.git</code><br/>

Sigma-FP has been released as a ROS package and works with Python 2.7. To install requirements, execute:

<code>cd Sigma-SP</code><br/>
<code>pip install -r requirements.txt</code><br/>

Build ROS workspace:

<code>cd ~/your_ros_workspace</code><br/>
<code>cd catkin_make</code>

Sigma-FP requires a per-pixel semantic segmentation network to run. We have employed Detectron2, but any other per-pixel semantic segmentation network can be used, although Sigma-FP code will need to be slightly adapted. In case you wish to use Detectron2, we have released our adaptation in the following repository: [detectron2_ros_probs](https://github.com/josematez/detectron2_ros_probs). The installation instructions are available in the repository.

### Configuration

### How to Run


