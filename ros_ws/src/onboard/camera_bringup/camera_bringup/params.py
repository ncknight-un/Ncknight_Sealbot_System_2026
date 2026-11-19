"""Load the Sealbot camera parameter file for use in launch files."""

import os

import yaml
from ament_index_python.packages import get_package_share_directory


def config_dir():
    return os.path.join(get_package_share_directory('camera_bringup'), 'config')


def default_params_file():
    return os.path.join(config_dir(), 'sealbot_camera.yaml')


def _resolve(path):
    """Bare filename -> camera_bringup/config/<name>; absolute paths unchanged."""
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(config_dir(), path)


def load_camera_params(params_file=None):
    """Return the camera node parameters with file paths fully resolved.

    settings_file becomes an absolute path and camera_info_url a file:// URL,
    so the dict can be passed straight to the vimbax_camera node.
    """
    params_file = params_file or default_params_file()
    with open(params_file) as f:
        data = yaml.safe_load(f)
    # Accept '/**', a node name, or a namespaced node name as the top-level key.
    params = dict(next(iter(data.values()))['ros__parameters'])

    if 'settings_file' in params:
        params['settings_file'] = _resolve(params['settings_file'])
    if 'camera_info_url' in params:
        url = params['camera_info_url']
        if '://' not in url:
            url = 'file://' + _resolve(url)
        params['camera_info_url'] = url
    return params


def camera_info_path(params):
    url = params['camera_info_url']
    if not url.startswith('file://'):
        raise ValueError(f'camera_info_url must be a file:// URL, got {url!r}')
    return url[len('file://'):]


def image_size(params):
    """(width, height) from the camera_info file referenced by params."""
    with open(camera_info_path(params)) as f:
        info = yaml.safe_load(f)
    return int(info['image_width']), int(info['image_height'])
