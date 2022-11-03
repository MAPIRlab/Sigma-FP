# <p align="center"> Sigma-FP: Robot Mapping of 3D Floor Plans with an RGB-D Camera under Uncertainty </p>

<p align="center"> <a href="https://mapir.isa.uma.es/mapirwebsite/?p=1792">Jose-Luis Matez-Bandera</a><sup>1</sup>, <a href="https://mapir.isa.uma.es/mapirwebsite/?p=1438">Javier Monroy</a><sup>1</sup> and <a href="http://mapir.isa.uma.es/jgonzalez">Javier Gonzalez-Jimenez</a><sup>1</sup> </p>

<p align="center"> <sup>1</sup> Machine Perception and Intelligent Robotics (MAPIR) Group,<br/> Malaga Institute for Mechatronics Engineering and Cyber-Physical Systems (IMECH.UMA).<br/> University of Malaga. Spain. </p>

### Content
<p align="center"> <a href="#citation">Citation</a>&nbsp;&nbsp;&nbsp;<a href="#installationandrequirements">Installation&nbsp;and&nbsp;Requirements</a>&nbsp;&nbsp;&nbsp;<a href="#configuration">Configuration</a>&nbsp;&nbsp;&nbsp;<a href="#howtorun">How&nbsp;to&nbsp;Run</a>&nbsp;&nbsp;&nbsp;<a href="#datasets">Datasets</a></p></p>

### Citation
<pre><code>@article{matez2022,  
    title={Sigma-FP: Robot Mapping of 3D Floor Plans with an RGB-D Camera under Uncertainty},  
    author={Matez-Bandera, Jose-Luis and Monroy, Javier and Gonzalez-Jimenez, Javier},  
    year={2022},
    journal={IEEE Robotics and Automation Letters},
    journal={IEEE Robotics and Automation Letters},
    year={2022},
    publisher={IEEE}
    }
</code></pre>

### Installation&nbsp;and&nbsp;Requirements

Clone the repository in the /src directory of your ROS workspace:

<code>git clone https://github.com/MAPIRlab/Sigma-FP.git</code><br/>

Sigma-FP has been released as a ROS package and works with Python 2.7. To install requirements, execute:

<code>cd Sigma-SP</code><br/>
<code>pip install -r requirements.txt</code><br/>

Additionally, it is required to install the following ROS packages:

- [cv_bridge](http://wiki.ros.org/cv_bridge)
- [tf2_ros](http://wiki.ros.org/tf2_ros)
- [tf](http://wiki.ros.org/tf)
- [message_filters](http://wiki.ros.org/message_filters)

Build ROS workspace:

<code>cd ~/your_ros_workspace</code><br/>
<code>cd catkin_make</code>

Sigma-FP requires a per-pixel semantic segmentation network to run. We have employed Detectron2, but any other per-pixel semantic segmentation network can be used, although Sigma-FP code will need to be slightly adapted. In case you wish to use Detectron2, we have released our adaptation in the following repository: [detectron2_ros_probs](https://github.com/josematez/detectron2_ros_probs). The installation instructions are available in the repository.

### Configuration

Sigma-FP parameters are configured using launch parameters. The configurable parameters are:

## Parameters
```bash
        # Name of the dataset to use (options: "RobotAtVirtualHome", "OpenLORIS", "Giraff" (this is for MAPIRlab) - leave empty for custom dataset)
        <param name="dataset" value="Giraff"/>
        # Topic where the RGB image is published
        <param name="topic_cameraRGB" value="camera_down/rgb/image_raw/compressed"/>
        # Topic where the Depth image is published
        <param name="topic_cameraDepth" value="/camera_down/depth/image"/>
        # Topic where the CNN results are published
        <param name="topic_result" value="ViMantic/Detections"/>
        # Topic where the CNN expects to receive the input image
        <param name="topic_republic" value="ViMantic/ToCNN"/>
        # Topic where the CNN publish the image including detections
        <param name="topic_cnn" value="detectron2_ros/result"/>
        # Debug option
        <param name="debug" value="false"/>
        # Image width
        <param name="image_width" value="640"/>
        # Image height
        <param name="image_height" value="480"/>
        # Intrinsic parameters of the camera
        <param name="camera_cx" value="318.2640075683594"/>
        <param name="camera_cy" value="237.88600158691406"/>
        <param name="camera_fx" value="510.3919982910156"/>
        <param name="camera_fy" value="510.3919982910156"/>
        # Max range of the depth camera
        <param name="camera_depth_max_range" value="10.0"/>
        # Number of desired point to downsample each input point cloud
        <param name="points_in_pcd" value="4000"/>
        # Minimum number of points to accept a planar patch as a candidate
        <param name="min_points_plane" value="100"/>
        # Minimum width (in meters) of a planar patch to accept it as a candidate
        <param name="min_plane_width" value="0.6"/>
        # Minimum number of pixels to consider a region as a opening in the image plane
        <param name="min_px_opening" value="8000"/>
        # Threshold for the statistical distance of Bhattacharyya
        <param name="bhattacharyya_threshold" value="7"/>
        # Threshold for the minimum euclidean distance between walls (in meters)
        <param name="euclidean_threshold" value="0.3"/>
        # Epsilon for DBSCAN of the azimuth angle of the plane (in radians)
        <param name="eps_alpha" value="1.0"/>
        # Epsilon for DBSCAN of the elevation angle of the plane (in radians)
        <param name="eps_beta" value="10.0"/>
        # Epsilon for DBSCAN of the plane-to-origin distance (in meters)
        <param name="eps_dist" value="0.02"/>
       
```

### How&nbsp;to&nbsp;Run

First, run the semantic segmentation neural network. For example, if you are using our recommended neural network ([Detectron2](https://github.com/josematez/detectron2_ros_probs)), first you need to activate the virtual environment:

<code>workon detectron2_ros</code><br/>

Then, execute the launch file:

<code>roslaunch detectron2_ros panoptic_detectron2_ros.launch</code><br/>

Once the semantic segmentation network is ready, you can run Sigma-FP as follows*:

<code>roslaunch sigmafp MAPIRlab.launch</code><br/>

<i>*Note that it is an example with the MAPIRlab dataset. For custom data, please create a launch file following the examples given in the launch directory.</i>

### Datasets

If you are interested in reproducing results, please contact us at <a>josematez@uma.es</a> to provide the employed datasets.
