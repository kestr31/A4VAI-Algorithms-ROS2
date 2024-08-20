# Library

import matplotlib.pyplot as plt
import numpy as np

# Library for ros2
import rclpy
from rclpy.node import Node

# Library for custom message
from custom_msgs.msg import GlobalWaypointSetpoint, LocalWaypointSetpoint

# Library for px4 message
from px4_msgs.msg import VehicleLocalPosition

# submodule for initial variables
from .initVar import *


class Plotter(Node):
    def __init__(self):
        super().__init__("plotter")

        setInitialVariables(self)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region Subscriber

        # Subscriber for global waypoint setpoint from controller
        self.global_waypoint_subscriber = self.create_subscription(
            GlobalWaypointSetpoint,
            "/global_waypoint_setpoint",
            self.global_waypoint_callback,
            10,
        )

        # Subscriber for local waypoint setpoint from path planning
        self.local_waypoint_subscriber = self.create_subscription(
            LocalWaypointSetpoint,
            "/local_waypoint_setpoint_from_PP",
            self.local_waypoint_callback,
            10,
        )

        # Subscriber for vehicle local position from px4
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            self.qos_profile,
        )
        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # region timer

        # update plot timer
        period_update_plot = 0.1
        self.timer = self.create_timer(period_update_plot, self.update_plot)

        # endregion
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # region Callback functions

    # global waypoint callback function
    def global_waypoint_callback(self, msg):
        self.start_global_waypoint = msg.start_point
        self.goal_global_waypoint = msg.goal_point
        self.global_waypoint_set = True

    # local waypoint callback function
    def local_waypoint_callback(self, msg):
        self.waypoint_x = msg.waypoint_x
        self.waypoint_y = msg.waypoint_y
        self.waypoint_z = msg.waypoint_z
        self.local_waypoint_set = msg.path_planning_complete

    # vehicle local position callback function
    def vehicle_local_position_callback(self, msg):
        # convert data to list
        vehicle_x = np.array(msg.x)
        vehicle_y = np.array(msg.y)
        vehicle_z = np.array(-msg.z)

        # append to list
        self.vehicle_x = np.append(self.vehicle_x, vehicle_x).flatten()
        self.vehicle_y = np.append(self.vehicle_y, vehicle_y).flatten()
        self.vehicle_z = np.append(self.vehicle_z, vehicle_z).flatten()

    # endregion
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Draw the updated plot
    def update_plot(self):

        # check if global and local waypoints are set
        if self.global_waypoint_set == True and self.local_waypoint_set == True:

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 1 full trajectory

            # Clear the previous plot
            self.ax1.clear()

            # Plot global waypoints with blue color
            self.ax1.scatter(
                [self.start_global_waypoint[0], self.goal_global_waypoint[0]],
                [self.start_global_waypoint[2], self.goal_global_waypoint[2]],
                color="blue",
                label="Global Waypoint",
                s=70,
            )

            # Plot local waypoints with red color
            self.ax1.scatter(
                self.waypoint_x,
                self.waypoint_y,
                color="red",
                label="Local Waypoints",
                s=6,
            )
            self.ax1.plot(
                self.waypoint_x,
                self.waypoint_y,
                color="red",
                linewidth=1,
            )

            # Plot vehicle positions
            if len(self.vehicle_x) > 0:
                self.ax1.plot(
                    self.vehicle_x,
                    self.vehicle_y,
                    color="green",
                    label="Vehicle Position",
                    linewidth=1,
                )

            # set the title, x and y labels
            self.ax1.set_title("full trajectory")
            self.ax1.set_xlabel("X Coordinate")
            self.ax1.set_ylabel("Y Coordinate")
            self.ax1.legend()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 2 vihicle fixed trajectory

            # Clear the previous plot
            self.ax2.clear()

            # Plot global waypoints with blue color
            self.ax2.scatter(
                [self.start_global_waypoint[0], self.goal_global_waypoint[0]],
                [self.start_global_waypoint[2], self.goal_global_waypoint[2]],
                color="blue",
                label="Global Waypoint",
                s=70,
            )

            # Plot local waypoints with red color
            self.ax2.scatter(
                self.waypoint_x,
                self.waypoint_y,
                color="red",
                label="Local Waypoints",
                s=6,
            )
            self.ax2.plot(
                self.waypoint_x,
                self.waypoint_y,
                color="red",
                linewidth=4,
            )

            if len(self.vehicle_x) > 0:
                self.ax2.plot(
                    self.vehicle_x,
                    self.vehicle_y,
                    color="green",
                    label="Vehicle Position",
                    linewidth=4,
                )
                # 드론의 위치에 따라 시점 고정
                x_center = self.vehicle_x[-1]
                y_center = self.vehicle_y[-1]

                margin = 10  # 드론의 주변을 보기 위한 마진

                self.ax2.set_xlim([x_center - margin, x_center + margin])
                self.ax2.set_ylim([y_center - margin, y_center + margin])

            # set the title, x and y labels
            self.ax2.set_title("Vihicle Position")
            self.ax2.set_xlabel("X Coordinate")
            self.ax2.set_ylabel("Y Coordinate")
            self.ax2.legend()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 3 altutude

            # # Clear the previous plot
            # self.ax3.clear()

            # # Plot global waypoints with blue color
            # self.ax3.scatter(
            #     [self.start_global_waypoint[0], self.goal_global_waypoint[0]],
            #     [self.start_global_waypoint[2], self.goal_global_waypoint[2]],
            #     [self.start_global_waypoint[1], self.goal_global_waypoint[1]],
            #     color="blue",
            #     label="Global Waypoint",
            #     s=70,
            # )

            # # Plot local waypoints with red color
            # self.ax3.scatter(
            #     self.waypoint_x,
            #     self.waypoint_y,
            #     self.waypoint_z,
            #     color="red",
            #     label="Local Waypoints",
            #     s=6,
            # )
            # self.ax3.plot(
            #     self.waypoint_x,
            #     self.waypoint_y,
            #     self.waypoint_z,
            #     color="red",
            #     linewidth=4,
            # )

            # if len(self.vehicle_x) > 0:
            #     self.ax3.plot(
            #         self.vehicle_x,
            #         self.vehicle_y,
            #         self.vehicle_z,
            #         color="green",
            #         label="Vehicle Position",
            #         linewidth=4,
            #     )
            #     # 드론의 위치에 따라 시점 고정
            #     x_center = self.vehicle_x[-1]
            #     y_center = self.vehicle_y[-1]
            #     z_center = self.vehicle_z[-1]

            #     margin = 10  # 드론의 주변을 보기 위한 마진

            #     self.ax3.set_xlim([x_center - margin, x_center + margin])
            #     self.ax3.set_ylim([y_center - margin, y_center + margin])
            #     self.ax3.set_zlim([z_center - margin, z_center + margin])
            #     self.ax3.view_init(elev=0, azim=-45)

            # # set the title, x and y labels
            # self.ax3.set_title("Vihicle Position")
            # self.ax3.set_xlabel("X Coordinate")
            # self.ax3.set_ylabel("Y Coordinate")
            # self.ax3.set_zlabel("Z Coordinate")
            # self.ax3.legend()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 3 altutude

            # Clear the previous plot
            self.ax3.clear()

            # Plot global waypoints with blue color
            self.ax3.scatter(
                [self.start_global_waypoint[0], self.goal_global_waypoint[0]],
                [self.start_global_waypoint[1], self.goal_global_waypoint[1]],
                color="blue",
                label="Global Waypoint",
                s=70,
            )

            # Plot local waypoints with red color
            self.ax3.scatter(
                self.waypoint_x,
                self.waypoint_z,
                color="red",
                label="Local Waypoints",
                s=6,
            )
            self.ax3.plot(
                self.waypoint_x,
                self.waypoint_z,
                color="red",
                linewidth=4,
            )

            if len(self.vehicle_x) > 0:
                self.ax3.plot(
                    self.vehicle_x,
                    self.vehicle_z,
                    color="green",
                    label="Vehicle Position",
                    linewidth=4,
                )
                # 드론의 위치에 따라 시점 고정
                x_center = self.vehicle_x[-1]
                z_center = self.vehicle_z[-1]

                margin = 10  # 드론의 주변을 보기 위한 마진

                self.ax3.set_xlim([x_center - margin, x_center + margin])
                self.ax3.set_ylim([z_center - margin, z_center + margin])

            # set the title, x and y labels
            self.ax3.set_title("Altitude X-Z plane")
            self.ax3.set_xlabel("X Coordinate")
            self.ax3.set_ylabel("Z Coordinate")
            self.ax3.legend()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # region Plot 4 altutude

            # Clear the previous plot
            self.ax4.clear()

            # Plot global waypoints with blue color
            self.ax4.scatter(
                [self.start_global_waypoint[2], self.goal_global_waypoint[2]],
                [self.start_global_waypoint[1], self.goal_global_waypoint[1]],
                color="blue",
                label="Global Waypoint",
                s=70,
            )

            # Plot local waypoints with red color
            self.ax4.scatter(
                self.waypoint_y,
                self.waypoint_z,
                color="red",
                label="Local Waypoints",
                s=6,
            )
            self.ax4.plot(
                self.waypoint_y,
                self.waypoint_z,
                color="red",
                linewidth=4,
            )

            if len(self.vehicle_x) > 0:
                self.ax4.plot(
                    self.vehicle_y,
                    self.vehicle_z,
                    color="green",
                    label="Vehicle Position",
                    linewidth=4,
                )
                # 드론의 위치에 따라 시점 고정
                y_center = self.vehicle_y[-1]
                z_center = self.vehicle_z[-1]

                margin = 10  # 드론의 주변을 보기 위한 마진

                self.ax4.set_xlim([y_center - margin, y_center + margin])
                self.ax4.set_ylim([z_center - margin, z_center + margin])

            # set the title, x and y labels
            self.ax4.set_title("Altitude Y-Z plane")
            self.ax4.set_xlabel("Y Coordinate")
            self.ax4.set_ylabel("Z Coordinate")
            self.ax4.legend()

            # endregion
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = Plotter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
