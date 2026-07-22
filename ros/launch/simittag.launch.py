from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("simittag_ros"), "config", "params.yaml"]
                ),
                description="Path to the simittag parameter file",
            ),
            Node(
                package="simittag_ros",
                executable="simittag_node",
                name="simittag",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
