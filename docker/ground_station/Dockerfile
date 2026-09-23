FROM ros:humble

RUN apt update && apt install -y \
    ros-humble-mavros \
    ros-humble-mavros-extras

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    ros2 run mavros install_geographiclib_datasets.sh"

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "alias humble-source='source /humble_ws/install/setup.bash'" >> /root/.bashrc

WORKDIR /humble_ws
