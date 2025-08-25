import argparse
import time
import numpy as np
import open3d as o3d
import pyspacemouse

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_depth_image_to_pointcloud,
)
from robo_manip_baselines.common.utils.Vision3dUtils import (
    crop_pointcloud_bb,
    downsample_pointcloud_fps,
    rotate_pointcloud,
)


class VisualizePointCloud:
    def __init__(self):
        self.setup_args()

    def setup_args(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("rmb_path", type=str)
        parser.add_argument("--camera_name", type=str, default="front")
        parser.add_argument("--skip", type=int, default=6)
        parser.add_argument(
            "--min_bound", type=float, nargs=3, default=[-0.4, -0.4, -0.4]
        )
        parser.add_argument("--max_bound", type=float, nargs=3, default=[1.0, 1.0, 1.0])
        parser.add_argument("--num_points", type=int, default=512)
        parser.add_argument("--roll_pitch_yaw", type=float, nargs=3, default=[0, 0, 0])
        self.args = parser.parse_args()

    def run(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.get_render_option().point_size = 10.0

        self.quit_flag = False
        self.time_idx = 0
        self.pointcloud_geom = None
        self.bbox_geom = None
        self.cam_params = None

        self.min_bound = np.array(self.args.min_bound)
        self.max_bound = np.array(self.args.max_bound)
        self.rpy = np.array(self.args.roll_pitch_yaw)
        self.downsample_enabled = False
        self.spacemouse = pyspacemouse.open(dof_callback=self.modify_bound_with_spacemouse)

        print("[Keys] <-/->: time step | Esc: exit | O: toggle downsample")
        print("E/D: min_x +/- | R/F: max_x +/-")
        print("T/G: min_y +/- | Y/H: max_y +/-")
        print("U/J: min_z +/- | I/K: max_z +/-")
        print("Z/X: roll +/- | C/V: pitch +/- | B/N: yaw +/-")
        print("3D mouse: parallel movement and rotation")

        with RmbData(self.args.rmb_path) as rmb_data:
            self.rgb_image_seq = rmb_data[
                DataKey.get_rgb_image_key(self.args.camera_name)
            ][:]
            self.depth_image_seq = rmb_data[
                DataKey.get_depth_image_key(self.args.camera_name)
            ][:]
            self.fov = rmb_data.attrs[
                DataKey.get_depth_image_key(self.args.camera_name) + "_fovy"
            ]
            self.max_time_idx = len(self.rgb_image_seq) - 1

            self.register_keys()

            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.2, origin=[0, 0, 0]
            )
            self.vis.add_geometry(coord_frame)

            self.update_pointcloud()

            while not self.quit_flag:
                self.state = self.spacemouse.read()
                self.vis.poll_events()
                self.vis.update_renderer()
                time.sleep(0.01)

        self.vis.destroy_window()

    def modify_bound_with_spacemouse(self, state):
        self.min_bound[0] += state.y * 0.02
        self.max_bound[0] += state.y * 0.02
        self.min_bound[1] += state.z * -0.02
        self.max_bound[1] += state.z * -0.02
        self.min_bound[2] += state.x * -0.02
        self.max_bound[2] += state.x * -0.02
        self.rpy[0] += state.roll
        self.rpy[1] += state.pitch
        self.rpy[2] += state.yaw

        self.update_pointcloud()

    def register_keys(self):
        # Time navigation
        self.vis.register_key_action_callback(262, self.right_callback)
        self.vis.register_key_action_callback(263, self.left_callback)
        self.vis.register_key_action_callback(256, self.esc_callback)

        # Bound control
        self.vis.register_key_action_callback(
            ord("E"), lambda v, a, m: self.modify_bound_with_modifiers("min", 0, 0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("D"), lambda v, a, m: self.modify_bound_with_modifiers("min", 0, -0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("R"), lambda v, a, m: self.modify_bound_with_modifiers("max", 0, 0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("F"), lambda v, a, m: self.modify_bound_with_modifiers("max", 0, -0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("T"), lambda v, a, m: self.modify_bound_with_modifiers("min", 1, 0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("G"), lambda v, a, m: self.modify_bound_with_modifiers("min", 1, -0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("Y"), lambda v, a, m: self.modify_bound_with_modifiers("max", 1, 0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("H"), lambda v, a, m: self.modify_bound_with_modifiers("max", 1, -0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("U"), lambda v, a, m: self.modify_bound_with_modifiers("min", 2, 0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("J"), lambda v, a, m: self.modify_bound_with_modifiers("min", 2, -0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("I"), lambda v, a, m: self.modify_bound_with_modifiers("max", 2, 0.05, a, m)
        )
        self.vis.register_key_action_callback(
            ord("K"), lambda v, a, m: self.modify_bound_with_modifiers("max", 2, -0.05, a, m)
        )

        # Rotation control
        self.vis.register_key_action_callback(
            ord("Z"), lambda v, a, m: self.modify_rpy_with_modifiers(0, 5, a, m)
        )
        self.vis.register_key_action_callback(
            ord("X"), lambda v, a, m: self.modify_rpy_with_modifiers(0, -5, a, m)
        )
        self.vis.register_key_action_callback(
            ord("C"), lambda v, a, m: self.modify_rpy_with_modifiers(1, 5, a, m)
        )
        self.vis.register_key_action_callback(
            ord("V"), lambda v, a, m: self.modify_rpy_with_modifiers(1, -5, a, m)
        )
        self.vis.register_key_action_callback(
            ord("B"), lambda v, a, m: self.modify_rpy_with_modifiers(2, 5, a, m)
        )
        self.vis.register_key_action_callback(
            ord("N"), lambda v, a, m: self.modify_rpy_with_modifiers(2, -5, a, m)
        )

        self.vis.register_key_action_callback(ord("O"), self.toggle_downsample)

    def get_modifier_step(self, base_step, action, mods):
        # mods (bitmask): 1=shift, 2=ctrl, 4=alt
        if mods & 2:  # Ctrl
            return base_step * 0.2
        elif mods & 4:  # Alt
            return base_step * 0.04
        else:
            return base_step

    def modify_bound_with_modifiers(self, which, axis, base_delta, action, mods):
        if action != 1:  # only on key-down
            return False
        delta = self.get_modifier_step(base_delta, action, mods)
        if which == "min":
            self.min_bound[axis] += delta
        else:
            self.max_bound[axis] += delta
        self.update_pointcloud()
        return False

    def modify_rpy_with_modifiers(self, axis, base_delta_deg, action, mods):
        if action != 1:
            return False
        delta = self.get_modifier_step(base_delta_deg, action, mods)
        self.rpy[axis] += delta
        self.update_pointcloud()
        return False

    def update_pointcloud(self):
        if not (0 <= self.time_idx <= self.max_time_idx):
            return

        rgb_image = self.rgb_image_seq[self.time_idx]
        depth_image = self.depth_image_seq[self.time_idx]
        result = convert_depth_image_to_pointcloud(depth_image, self.fov, rgb_image)
        if result is None or not isinstance(result, (tuple, list)) or len(result) != 2:
            print(f"[Warning] Invalid pointcloud at time index {self.time_idx}")
            return
        pointcloud = np.concatenate(result, axis=1)

        rotmat = self.euler_to_rotation_matrix(self.rpy)
        pointcloud = rotate_pointcloud(pointcloud, rotmat)
        pointcloud = crop_pointcloud_bb(pointcloud, self.min_bound, self.max_bound)
        if self.downsample_enabled:
            pointcloud = downsample_pointcloud_fps(pointcloud, self.args.num_points)

        view_ctrl = self.vis.get_view_control()
        if self.pointcloud_geom is not None:
            self.cam_params = view_ctrl.convert_to_pinhole_camera_parameters()
            self.vis.remove_geometry(self.pointcloud_geom)
        if self.bbox_geom is not None:
            self.vis.remove_geometry(self.bbox_geom)

        self.pointcloud_geom = o3d.geometry.PointCloud()
        self.pointcloud_geom.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
        colors = pointcloud[:, 3:] if pointcloud.shape[1] >= 6 else np.zeros_like(pointcloud[:, :3])
        self.pointcloud_geom.colors = o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(self.pointcloud_geom)

        self.bbox_geom = self.create_bounding_box(self.min_bound, self.max_bound)
        self.vis.add_geometry(self.bbox_geom)

        if self.cam_params is None:
            center = 0.5 * (self.min_bound + self.max_bound)
            view_ctrl.set_lookat(center.tolist())
            view_ctrl.set_front([0.0, 0.0, -1.0])
            view_ctrl.set_up([0.0, -1.0, 0.0])
            view_ctrl.set_zoom(0.15)
        else:
            view_ctrl.convert_from_pinhole_camera_parameters(self.cam_params)

    def create_bounding_box(self, min_b, max_b):
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
        line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
        line_set.paint_uniform_color([1.0, 0.0, 0.0])
        return line_set

    def euler_to_rotation_matrix(self, rpy_deg):
        r, p, y = np.deg2rad(rpy_deg)
        Rx = np.array(
            [[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]]
        )
        Ry = np.array(
            [[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]]
        )
        Rz = np.array(
            [[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]]
        )
        return Rz @ Ry @ Rx

    def toggle_downsample(self, vis, action, mods):
        if action != 1:
            return False
        self.downsample_enabled = not self.downsample_enabled
        self.update_pointcloud()
        return False

    def right_callback(self, vis, action, mods):
        if action != 1:
            return False
        self.time_idx = min(self.time_idx + self.args.skip, self.max_time_idx)
        self.update_pointcloud()
        return False

    def left_callback(self, vis, action, mods):
        if action != 1:
            return False
        self.time_idx = max(self.time_idx - self.args.skip, 0)
        self.update_pointcloud()
        return False

    def esc_callback(self, vis, action, mods):
        if action != 1:
            return False
        print("---bounding box parameter---")
        print(f"[roll pitch yaw] : {self.rpy}")
        print(f"[min_x min_y min_z] : {self.min_bound}")
        print(f"[max_x max_y max_z] : {self.max_bound}")
        self.quit_flag = True
        return False


if __name__ == "__main__":
    vis_pc = VisualizePointCloud()
    vis_pc.run()
