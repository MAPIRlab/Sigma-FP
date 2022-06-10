# 3D-Floorplan-Reconstruction

This repository corresponds to the work entitled "Robot Mapping of 3D Floorplans with an RGB-D camera" and submitted to IEEE Robotics and Automation Letters and 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2022). The code will be released if the work is accepted.

### Video

<a href="http://www.youtube.com/watch?feature=player_embedded&v=GkjAFzaJcjs" target="_blank">
 <img src="http://img.youtube.com/vi/GkjAFzaJcjs/mqdefault.jpg" alt="Watch the video" width="240" height="180" border="100" />
</a>

### Abstract

This work presents a novel 3D reconstruction method to obtain the floorplan of a multi-room environment from a sequence of RGB-D images captured by a wheeled mobile robot. For each input image, the planar patches of visible walls are extracted and subsequently characterized by a multivariate Gaussian distribution in the convenient Plane Parameter Space. Then, accounting for the probabilistic nature of the robot localization, we transform and combine the planar patches from the camera frame into a 3D global model, where the planar patches include both the plane estimation uncertainty and the propagation of the robot pose uncertainty. Additionally, leveraging computer vision techniques based on depth information, we detect openings (doors and windows) in the scene, which are also included in the 3D global model to enrich the level of detail. Experiments demonstrate that our method outperforms a state-of-the-art point cloud-based method, both in time and level of error, while also relaxing the common assumptions imposed to the room geometry.
