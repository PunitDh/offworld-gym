#!/usr/bin/env bash

# this assumes the script is executed from under `scripts` directory
# (should we move `install.sh` to the repository root? will be cleaner)
cd ..
export OFFWORLD_GYM_ROOT=`pwd`

# make sure we have Python 3.5
sudo apt install -y python3.5 python3.5-dev

# create virtual environment
sudo apt install -y virtualenv
mkdir ~/ve
virtualenv -p python3.5 ~/ve/py35gym
source ~/ve/py35gym/bin/activate
pip install --upgrade pip

echo "Virtual environment set up done."
# (we will drop this)
#echo "source ~/ve/py36/bin/activate" >> ~/.bashrc
#source ~/.bashrc

# intall Python packages
pip install numpy
pip install tensorflow-gpu
pip install keras
pip install opencv-python
pip install catkin_pkg
pip install empy
pip install defusedxml
pip install rospkg
pip install matplotlib
pip install netifaces
pip install regex
pip install psutil
pip install gym
pip install python-socketio
cd $OFFWORLD_GYM_ROOT
pip install -e .
echo "Python packages installed."

# install additional ROS packages
sudo apt install -y ros-kinetic-grid-map ros-kinetic-frontier-exploration ros-kinetic-ros-controllers ros-kinetic-rospack libignition-math2-dev python3-tk

# build Python 3.5 version of catkin *without* installing it system-wide
mkdir $OFFWORLD_GYM_ROOT/assets
cd $OFFWORLD_GYM_ROOT/assets
echo "Building catkin here: `pwd`."
git clone https://github.com/ros/catkin.git -b kinetic-devel
cd $OFFWORLD_GYM_ROOT/assets/catkin
mkdir build && cd build && cmake .. && make
echo "Catkin build for Python 3.5 complete."

# prepare for building the workspace
cd /usr/lib/x86_64-linux-gnu
sudo ln -s libboost_python-py35.so libboost_python3.so

# build ROS workspace
cd $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src
$OFFWORLD_GYM_ROOT/assets/catkin/bin/catkin_init_workspace

git clone https://github.com/ros/xacro.git -b kinetic-devel
git clone https://github.com/ros/ros.git -b kinetic-devel
git clone https://github.com/ros/ros_comm.git -b kinetic-devel
git clone https://github.com/ros/common_msgs.git -b indigo-devel
git clone https://github.com/ros/catkin.git -b kinetic-devel
git clone https://github.com/ros/ros_comm_msgs.git -b indigo-devel
git clone https://github.com/ros/gencpp.git -b indigo-devel
git clone https://github.com/jsk-ros-pkg/geneus.git -b master
git clone https://github.com/ros/genlisp.git -b groovy-devel
git clone https://github.com/ros/genmsg.git -b indigo-devel
git clone https://github.com/ros/genpy.git -b kinetic-devel
git clone https://github.com/RethinkRobotics-opensource/gennodejs.git -b kinetic-devel
git clone https://github.com/ros/std_msgs.git -b groovy-devel
git clone https://github.com/ros/geometry.git -b indigo-devel
git clone https://github.com/ros/geometry2.git -b indigo-devel
git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b kinetic-devel
git clone https://github.com/ros-controls/ros_control.git -b kinetic-devel
git clone https://github.com/ros/dynamic_reconfigure.git -b master
git clone https://github.com/offworld-projects/offworld_rosbot_description.git -b kinetic-devel
git clone https://github.com/ros-perception/vision_opencv.git -b kinetic

cd ..
$OFFWORLD_GYM_ROOT/assets/catkin/bin/catkin_make -j1

echo "ROS dependencies build complete."

# integrate the new environment into the system
#echo "source ~/ve/py35gym/bin/activate" >> ~/.bashrc
#echo "unset PYTHONPATH" >> ~/.bashrc
#echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
#echo "source $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/devel/setup.bash --extend" >> ~/.bashrc
#echo "export GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH" >> ~/.bashrc
#echo "export PYTHONPATH=~/ve/py35gym/lib/python3.5/site-packages:$PYTHONPATH" >> ~/.bashrc

# update to gazebo 7.13
# http://answers.gazebosim.org/question/18934/kinect-in-gazebo-not-publishing-topics/
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
sudo apt install wget
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install gazebo7
sudo apt-get install libgazebo7-dev

printf "Installation complete.\n\n"
printf "To test Real environment:\n\t(add instructions here)\n\n"
printf "To test Sim environment open two terminals and run:\n\t1. $OFFWORLD_GYM_ROOT/scripts/start_sim.sh\n\t2. $OFFWORLD_GYM_ROOT/scripts/start_gzclient.sh\n\n"
