cmake_minimum_required(VERSION 3.22)
project(simittag_ros)

find_package(ament_cmake REQUIRED)

# The node is pure Rust; cargo does the real build. r2r finds the message
# packages through the sourced ROS environment (AMENT_PREFIX_PATH).
add_custom_target(cargo_build ALL
  COMMAND cargo build --release --target-dir ${CMAKE_CURRENT_BINARY_DIR}/cargo
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  USES_TERMINAL
)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/cargo/release/simittag_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
