# Diffusion Uncertainty Frontier based path Planner

This repository extends the HKUST-Aerial-Robotics [FUEL](https://github.com/HKUST-Aerial-Robotics/FUEL) planner by adding a diffusion-based sampling stage for frontier expansion. Instead of only following detected frontiers, our method samples extra points in free space around each frontier to reduce uncertainty and improve goal selection.

## Key Features
- **Diffusion Sampling**: Randomly sample points within a radius around each frontier to gather more information.
- **Frontier Expansion**: Merge diffusion samples with original FUEL frontiers and re-cluster to select higher-value goals.
- **Real-Time Map Updates**: Subscribe to live point clouds and update the occupancy map continuously.
- **ROS Nodes**: Separate nodes for diffusion, frontier expansion, and FUEL integration. Parameters can be tuned via launch files.
- **Optional GPR Integration**: Use Gaussian Process Regression to predict unknown regions if needed.

## Installation & Build
```bash
# 1) Create or go to your Catkin workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# 2) Clone this repo
git clone git@github.com:romaster93/DUF-Planner.git

# 4) Install dependencies
cd ~/catkin_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 5) Build
catkin build


Update will continue...
