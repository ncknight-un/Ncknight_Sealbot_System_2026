FROM ros:humble

# Install Pre-req tools & ROS dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \ 
    curl \
    python3-vcstool \
    python3-rosinstall-generator \
    python3-osrf-pycommon \
    ros-humble-desktop \
    ros-humble-rmw-cyclonedds-cpp \
    && add-apt-repository -y universe \
    && rm -rf /var/lib/apt/lists/*

# Pull and build MAVLINK and MAVROSin an isolated directory: 
WORKDIR /opt/ros_deps
RUN mkdir -p src 

RUN rosinstall_generator --format repos mavlink | tee /tmp/mavlink.repos && \
    rosinstall_generator --format repos --upstream mavros | tee /tmp/mavros.repos && \
    vcs import src < /tmp/mavlink.repos && \
    vcs import src < /tmp/mavros.repos

# Install underlying dependencies and GeographicLib datasets
RUN apt-get update && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y --rosdistro humble && \
    bash ./src/mavros/mavros/scripts/install_geographiclib_datasets.sh \
    && rm -rf /var/lib/apt/lists/*

# Buikd dependency packages into /opt/ros_deps/install
RUN . /opt/ros/humble/setup.sh && colcon build --packages-up-to mavros --install-base /opt/ros_deps/install 

# Install AprilTag detection and camera dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-apriltag-ros \
    ros-humble-apriltag-msgs \
    ros-humble-camera-info-manager \
    && rm -rf /var/lib/apt/lists/*

# Install Mrcal tools for camera calibration setup on Station PC:
Run apt-get update && apt-get install -y \
    mrcal \
    libmrcal-dev \
    python3-mrcal \
    mrgingham \ 
    && rm -rf /var/lib/apt/lists/*

# Set the Runtime Directory: 
# RUN mkdir -p /tmp/runtime-root && chmod 0700 /tmp/runtime-root
# ENV XDG_RUNTIME_DIR=/tmp/runtime-root

# Create the workspace directory structure
WORKDIR /ros_ws

# Configure environment automatic sourcing and workspace shortcuts
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /opt/ros_deps/install/setup.bash" >> /root/.bashrc
RUN echo "alias humble-source='source /ros_ws/install/setup.bash'" >> /root/.bashrc
