"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import json
import os
import pathlib
from pathlib import Path
import pickle
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot
from gear_sonic.utils.inference.realtime_trace import EventTracer

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TRACE = EventTracer("simulator")


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            self.camera_configs = {
                "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
            }

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen

        self.init_scene()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
            elif "right_hand" in name:
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)

        # HandCmd uses the Dex3 semantic order from G1SupplementalInfo rather
        # than MuJoCo's XML joint-tree order.  Keep feedback, PD targets, and
        # actuator assignment in that same order.
        hand_joint_order = (
            "thumb_0_joint",
            "thumb_1_joint",
            "thumb_2_joint",
            "index_0_joint",
            "index_1_joint",
            "middle_0_joint",
            "middle_1_joint",
        )
        self.left_hand_index = np.array(
            [self.mj_model.joint(f"left_hand_{name}").id for name in hand_joint_order]
        )
        self.right_hand_index = np.array(
            [self.mj_model.joint(f"right_hand_{name}").id for name in hand_joint_order]
        )

    def init_renderers(self):
        self.renderers = {}
        self.renderer_scene_options = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer
            scene_option = mujoco.MjvOption()
            for geom_group in camera_config.get("hidden_geom_groups", ()):
                if not 0 <= geom_group < len(scene_option.geomgroup):
                    raise ValueError(f"Invalid MuJoCo geom group: {geom_group}")
                scene_option.geomgroup[geom_group] = 0
            self.renderer_scene_options[camera_name] = scene_option

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            scene_option = self.renderer_scene_options[camera_name]
            if "params" in camera_config:
                renderer.update_scene(
                    self.mj_data,
                    camera=camera_config["params"],
                    scene_option=scene_option,
                )
            elif "mjcf_name" in camera_config:
                renderer.update_scene(
                    self.mj_data,
                    camera=camera_config["mjcf_name"],
                    scene_option=scene_option,
                )
            else:
                renderer.update_scene(
                    self.mj_data,
                    camera=camera_name,
                    scene_option=scene_option,
                )
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)


class BottleTaskEnv(DefaultEnv):
    """Deterministic bottle/apple benchmark using the published SONIC scene."""

    TABLE_X_BOUNDS = (0.10, 0.70)
    TABLE_ABS_Y_BOUND = 0.78

    # The source demonstrations and closed-loop evaluation use twelve fixed
    # placements on the robot's right half of the table.  Labels run down each
    # column (C1-C4, C5-C8, C9-C12), from near to far in x.  MuJoCo's camera
    # right axis is negative world y.
    SINGLE_BOTTLE_POSITIONS = np.array(
        [
            [x, y, 0.875]
            for y in (-0.07, -0.21, -0.35)
            for x in (0.21, 0.31, 0.41, 0.51)
        ]
    )

    RIGHT_HAND_BODIES = [
        "right_hand_thumb_0_link",
        "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
        "right_hand_middle_0_link",
        "right_hand_middle_1_link",
        "right_hand_index_0_link",
        "right_hand_index_1_link",
    ]
    LOWER_BODY_JOINTS = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "pnp_bottle",
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        config = config.copy()
        self.scenario = os.environ.get("GROOT_WBC_TASK_SCENARIO", "single_bottle")
        self.target = os.environ.get("GROOT_WBC_TASK_TARGET", "bottle")
        self.seed = int(os.environ.get("GROOT_WBC_TASK_SEED", "0"))
        self.metrics_path = os.environ.get("GROOT_WBC_TASK_METRICS_PATH")
        self.metrics_history_path = os.environ.get("GROOT_WBC_TASK_HISTORY_PATH")
        self.arm_file = os.environ.get("GROOT_WBC_TASK_ARM_FILE")
        self.reset_file = os.environ.get("GROOT_WBC_TASK_RESET_FILE")
        self.reset_applied = self.reset_file is None
        self.lock_lower_body = os.environ.get("GROOT_WBC_LOCK_LOWER_BODY", "0") != "0"
        waist_yaw_bounds = os.environ.get("GROOT_WBC_WAIST_YAW_BOUNDS_RAD", "")
        self.waist_yaw_bounds = None
        if waist_yaw_bounds:
            parsed_bounds = np.fromstring(waist_yaw_bounds, sep=",", dtype=float)
            if (
                parsed_bounds.shape != (2,)
                or not np.all(np.isfinite(parsed_bounds))
                or parsed_bounds[0] >= parsed_bounds[1]
            ):
                raise ValueError(
                    "GROOT_WBC_WAIST_YAW_BOUNDS_RAD must contain finite lower,upper bounds"
                )
            self.waist_yaw_bounds = parsed_bounds
        self.assisted_grasp_enabled = (
            os.environ.get("GROOT_WBC_ASSISTED_GRASP", "0") != "0"
        )
        self.assisted_grasp_close_threshold = float(
            os.environ.get("GROOT_WBC_ASSISTED_GRASP_CLOSE_THRESHOLD", "0.5")
        )
        self.assisted_grasp_release_threshold = float(
            os.environ.get("GROOT_WBC_ASSISTED_GRASP_RELEASE_THRESHOLD", "0.15")
        )
        self.assisted_grasp_contact_grace = float(
            os.environ.get("GROOT_WBC_ASSISTED_GRASP_CONTACT_GRACE_S", "0.75")
        )
        if not (
            0.0
            <= self.assisted_grasp_release_threshold
            < self.assisted_grasp_close_threshold
        ):
            raise ValueError(
                "Assisted-grasp thresholds must satisfy 0 <= release < close"
            )
        if self.assisted_grasp_contact_grace < 0.0:
            raise ValueError("Assisted-grasp contact grace must be non-negative")
        self.task_duration = float(os.environ.get("GROOT_WBC_TASK_DURATION_S", "90"))
        self.lift_height = float(os.environ.get("GROOT_WBC_TASK_LIFT_HEIGHT_M", "0.045"))
        self.hold_time = float(os.environ.get("GROOT_WBC_TASK_HOLD_TIME_S", "0.5"))

        if self.scenario == "single_bottle":
            config["ROBOT_SCENE"] = (
                "decoupled_wbc/control/robot_model/model_data/g1/pnp_bottle_43dof.xml"
            )
        elif self.scenario == "bottle_apple":
            config["ROBOT_SCENE"] = (
                "decoupled_wbc/control/robot_model/model_data/g1/"
                "pnp_bottle_apple_43dof.xml"
            )
        else:
            raise ValueError(f"Unsupported task scenario: {self.scenario}")

        if self.target not in {"bottle", "apple"}:
            raise ValueError(f"Unsupported task target: {self.target}")
        if self.target == "apple" and self.scenario != "bottle_apple":
            raise ValueError("The apple target requires the bottle_apple scenario")

        camera_configs = {
            "ego_view": {
                "height": 480,
                "width": 640,
                "mjcf_name": "egoview",
                # Robot visual meshes use group 1; collision-only duplicates use
                # group 0 and must not appear in the policy RGB observation.
                "hidden_geom_groups": (0,),
            },
        }
        super().__init__(
            config,
            env_name,
            camera_configs,
            onscreen,
            offscreen,
            enable_image_publish,
        )

        prearm_body_q = os.environ.get("GROOT_WBC_TASK_ROBOT_BODY_Q")
        self._prearm_body_q = None
        if prearm_body_q:
            self._prearm_body_q = np.fromstring(prearm_body_q, sep=",", dtype=float)
            if self._prearm_body_q.shape != (self.num_body_dof,) or not np.all(
                np.isfinite(self._prearm_body_q)
            ):
                raise ValueError(
                    "GROOT_WBC_TASK_ROBOT_BODY_Q must contain "
                    f"{self.num_body_dof} finite comma-separated coordinates"
                )

        self._place_objects()
        self.object_body_name = f"{self.target}_body"
        self.object_geom_name = self.target
        self.object_body = self.mj_model.body(self.object_body_name)
        self.object_geom = self.mj_model.geom(self.object_geom_name)
        self._assisted_grasp_equality_id = self.mj_model.equality(
            "assisted_grasp_weld"
        ).id
        self.initial_object_z = float(self.mj_data.xpos[self.object_body.id][2])
        self.max_object_z = self.initial_object_z
        self.contact_observed = False
        self.current_contact = False
        self._trace_contact_emitted = False
        self._trace_lift_emitted = False
        self._trace_terminal_status = None
        self.assisted_grasp_active = False
        self.assisted_grasp_activations = 0
        self._assisted_grasp_relative_position = None
        self._assisted_grasp_relative_quaternion = None
        self._last_right_hand_contact_wall_time = None
        self.mj_data.eq_active[self._assisted_grasp_equality_id] = 0
        self.lift_started_wall_time = None
        self.success = False
        self.wrong_object_lifted = False
        self.robot_falls = 0
        self.simulator_instabilities = 0
        self.simulator_unstable = False
        self.armed_at = None
        self.armed_wall_time = None
        self.simulator_elapsed = 0.0
        self._locked_root_qpos = None
        self._locked_lower_qpos = {}
        self._last_metrics_write = -1.0
        self._write_metrics(force=True)

    def _set_free_body_position(self, body_name: str, position: np.ndarray):
        body_id = self.mj_model.body(body_name).id
        joint_id = int(self.mj_model.body_jntadr[body_id])
        qpos_address = int(self.mj_model.jnt_qposadr[joint_id])
        self.mj_data.qpos[qpos_address : qpos_address + 3] = position
        self.mj_model.qpos0[qpos_address : qpos_address + 3] = position

    def _reset_free_body(self, body_name: str):
        body_id = self.mj_model.body(body_name).id
        joint_id = int(self.mj_model.body_jntadr[body_id])
        qpos_address = int(self.mj_model.jnt_qposadr[joint_id])
        dof_address = int(self.mj_model.jnt_dofadr[joint_id])
        self.mj_data.qpos[qpos_address : qpos_address + 7] = self.mj_model.qpos0[
            qpos_address : qpos_address + 7
        ]
        self.mj_data.qvel[dof_address : dof_address + 6] = 0.0

    def _maybe_reset_objects(self):
        if self.reset_applied or not self.reset_file or not Path(self.reset_file).exists():
            return
        self._place_objects()
        self._reset_free_body("bottle_body")
        if self.scenario == "bottle_apple":
            self._reset_free_body("apple_body")
        self._apply_prearm_robot_pose()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.initial_object_z = float(self.mj_data.xpos[self.object_body.id][2])
        self.max_object_z = self.initial_object_z
        self.contact_observed = False
        self.current_contact = False
        self._trace_contact_emitted = False
        self._trace_lift_emitted = False
        self._trace_terminal_status = None
        self.assisted_grasp_active = False
        self.assisted_grasp_activations = 0
        self._assisted_grasp_relative_position = None
        self._assisted_grasp_relative_quaternion = None
        self._last_right_hand_contact_wall_time = None
        self.mj_data.eq_active[self._assisted_grasp_equality_id] = 0
        self.lift_started_wall_time = None
        self.success = False
        self.wrong_object_lifted = False
        self.reset_applied = True
        _TRACE.emit(
            "objects_reset",
            object_z=self.initial_object_z,
            sim_time_s=float(self.mj_data.time),
        )
        self._write_metrics(force=True)

    def _apply_prearm_robot_pose(self):
        """Hold the simulator at the demonstration start state until arming."""
        if self._prearm_body_q is None:
            return
        body_qpos = self.body_joint_index + self.qpos_offset - 1
        body_qvel = self.body_joint_index + self.qvel_offset - 1
        self.mj_data.qpos[body_qpos] = self._prearm_body_q
        self.mj_data.qvel[body_qvel] = 0.0
        for hand_indices in (self.left_hand_index, self.right_hand_index):
            hand_qpos = hand_indices + self.qpos_offset - 1
            hand_qvel = hand_indices + self.qvel_offset - 1
            self.mj_data.qpos[hand_qpos] = 0.0
            self.mj_data.qvel[hand_qvel] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _place_objects(self):
        rng = np.random.default_rng(self.seed)
        if self.scenario == "single_bottle":
            position_override = os.environ.get("GROOT_WBC_TASK_BOTTLE_POSITION")
            if position_override:
                bottle_pos = np.fromstring(position_override, sep=",", dtype=float)
                if bottle_pos.shape != (3,) or not np.all(np.isfinite(bottle_pos)):
                    raise ValueError(
                        "GROOT_WBC_TASK_BOTTLE_POSITION must contain three finite "
                        "comma-separated coordinates"
                    )
            else:
                bottle_pos = self.SINGLE_BOTTLE_POSITIONS[
                    self.seed % len(self.SINGLE_BOTTLE_POSITIONS)
                ].copy()
            self._set_free_body_position("bottle_body", bottle_pos)
        else:
            jitter_x = 0.0 if self.seed == 0 else rng.uniform(-0.035, 0.035)
            jitter_y = 0.0 if self.seed == 0 else rng.uniform(-0.02, 0.02)
            self._set_free_body_position(
                "bottle_body", np.array([0.4 + jitter_x, -0.08 + jitter_y, 0.875])
            )
            self._set_free_body_position(
                "apple_body", np.array([0.4 - jitter_x, 0.08 - jitter_y, 0.842])
            )
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _is_armed(self) -> bool:
        if self.armed_at is not None:
            return True
        if self.arm_file is None or Path(self.arm_file).exists():
            if self.lock_lower_body:
                self._capture_lower_body_lock()
            self.armed_at = float(self.mj_data.time)
            self.armed_wall_time = time.monotonic()
            _TRACE.emit(
                "task_armed",
                sim_time_s=self.armed_at,
                initial_object_z=self.initial_object_z,
            )
            return True
        return False

    def _capture_lower_body_lock(self):
        self._locked_root_qpos = self.mj_data.qpos[:7].copy()
        for joint_name in self.LOWER_BODY_JOINTS:
            joint_id = self.mj_model.joint(joint_name).id
            qpos_address = int(self.mj_model.jnt_qposadr[joint_id])
            self._locked_lower_qpos[joint_name] = float(self.mj_data.qpos[qpos_address])

    def _apply_lower_body_lock(self):
        if self._locked_root_qpos is None:
            return
        self.mj_data.qpos[:7] = self._locked_root_qpos
        self.mj_data.qvel[:6] = 0.0
        for joint_name, position in self._locked_lower_qpos.items():
            joint_id = self.mj_model.joint(joint_name).id
            qpos_address = int(self.mj_model.jnt_qposadr[joint_id])
            dof_address = int(self.mj_model.jnt_dofadr[joint_id])
            self.mj_data.qpos[qpos_address] = position
            self.mj_data.qvel[dof_address] = 0.0
        if self.waist_yaw_bounds is not None:
            waist_yaw = self.mj_model.joint("waist_yaw_joint")
            qpos_address = int(self.mj_model.jnt_qposadr[waist_yaw.id])
            dof_address = int(self.mj_model.jnt_dofadr[waist_yaw.id])
            position = float(self.mj_data.qpos[qpos_address])
            constrained = float(
                np.clip(
                    position,
                    self.waist_yaw_bounds[0],
                    self.waist_yaw_bounds[1],
                )
            )
            self.mj_data.qpos[qpos_address] = constrained
            if constrained != position:
                self.mj_data.qvel[dof_address] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _right_hand_closure(self) -> float:
        joint_names = (
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        )
        positions = [
            float(
                self.mj_data.qpos[
                    self.mj_model.jnt_qposadr[self.mj_model.joint(name).id]
                ]
            )
            for name in joint_names
        ]
        return float(np.mean(positions))

    def _activate_assisted_grasp(self):
        """Capture the current object pose relative to the live wrist pose."""
        anchor = self.mj_model.body("right_wrist_yaw_link")
        anchor_position = self.mj_data.xpos[anchor.id].copy()
        anchor_rotation = self.mj_data.xmat[anchor.id].reshape(3, 3).copy()
        anchor_quaternion = self.mj_data.xquat[anchor.id].copy()
        object_position = self.mj_data.xpos[self.object_body.id].copy()
        object_quaternion = self.mj_data.xquat[self.object_body.id].copy()

        self._assisted_grasp_relative_position = anchor_rotation.T @ (
            object_position - anchor_position
        )
        inverse_anchor_quaternion = np.empty(4, dtype=float)
        relative_quaternion = np.empty(4, dtype=float)
        mujoco.mju_negQuat(inverse_anchor_quaternion, anchor_quaternion)
        mujoco.mju_mulQuat(
            relative_quaternion, inverse_anchor_quaternion, object_quaternion
        )
        self._assisted_grasp_relative_quaternion = relative_quaternion
        equality_data = self.mj_model.eq_data[self._assisted_grasp_equality_id]
        equality_data[:3] = 0.0
        equality_data[3:6] = self._assisted_grasp_relative_position
        equality_data[6:10] = self._assisted_grasp_relative_quaternion
        self.mj_data.eq_active[self._assisted_grasp_equality_id] = 1
        self.assisted_grasp_active = True
        self.assisted_grasp_activations += 1
        _TRACE.emit(
            "assisted_grasp_activated",
            activation=self.assisted_grasp_activations,
            hand_closure=self._right_hand_closure(),
            object_z=float(self.mj_data.xpos[self.object_body.id][2]),
        )
        print(
            "Assisted grasp latched after physical right-hand contact "
            f"(closure={self._right_hand_closure():.3f})",
            flush=True,
        )

    def sim_step(self):
        previous_sim_time = float(self.mj_data.time)
        super().sim_step()
        if self.reset_applied and self.armed_at is None:
            self._apply_prearm_robot_pose()
        current_sim_time = float(self.mj_data.time)
        state_is_finite = all(
            np.all(np.isfinite(values))
            for values in (self.mj_data.qpos, self.mj_data.qvel, self.mj_data.qacc)
        )
        if self.armed_at is not None:
            if current_sim_time + 1e-9 < previous_sim_time or not state_is_finite:
                self.simulator_instabilities += 1
                self.simulator_unstable = True
            else:
                self.simulator_elapsed += current_sim_time - previous_sim_time
        if self.lock_lower_body and self.armed_at is not None:
            self._apply_lower_body_lock()

    def _object_lifted(self, body_name: str, initial_z: float) -> bool:
        body_id = self.mj_model.body(body_name).id
        return float(self.mj_data.xpos[body_id][2]) >= initial_z + self.lift_height

    def update_reward(self):
        self._maybe_reset_objects()
        if not self._is_armed():
            self._write_metrics()
            return

        if self.simulator_unstable:
            self._write_metrics(force=True)
            return

        wall_time = time.monotonic()
        object_z = float(self.mj_data.xpos[self.object_body.id][2])
        self.max_object_z = max(self.max_object_z, object_z)
        physical_contact = check_contact(
            self.mj_model,
            self.mj_data,
            self.RIGHT_HAND_BODIES,
            self.object_body_name,
        )
        hand_closure = self._right_hand_closure()
        if physical_contact:
            self._last_right_hand_contact_wall_time = wall_time
        recent_contact = bool(
            self._last_right_hand_contact_wall_time is not None
            and wall_time - self._last_right_hand_contact_wall_time
            <= self.assisted_grasp_contact_grace
        )
        if (
            self.assisted_grasp_enabled
            and not self.assisted_grasp_active
            and recent_contact
            and hand_closure >= self.assisted_grasp_close_threshold
        ):
            self._activate_assisted_grasp()
        elif (
            self.assisted_grasp_active
            and hand_closure <= self.assisted_grasp_release_threshold
        ):
            self.mj_data.eq_active[self._assisted_grasp_equality_id] = 0
            self.assisted_grasp_active = False
            print(
                "Assisted grasp released after hand opening "
                f"(closure={hand_closure:.3f})",
                flush=True,
            )
        current_contact = physical_contact or self.assisted_grasp_active
        self.current_contact = current_contact
        self.contact_observed = self.contact_observed or current_contact
        lifted = object_z >= self.initial_object_z + self.lift_height
        if current_contact and not self._trace_contact_emitted:
            self._trace_contact_emitted = True
            _TRACE.emit(
                "contact_started",
                physical_contact=physical_contact,
                assisted_grasp_active=self.assisted_grasp_active,
                hand_closure=hand_closure,
                object_z=object_z,
            )
        if lifted and not self._trace_lift_emitted:
            self._trace_lift_emitted = True
            _TRACE.emit(
                "lift_started",
                object_z=object_z,
                lift_m=object_z - self.initial_object_z,
                contact=current_contact,
            )
        position = self.mj_data.xpos[self.object_body.id]
        object_in_bounds = (
            position[2] >= 0.2
            and self.TABLE_X_BOUNDS[0] <= position[0] <= self.TABLE_X_BOUNDS[1]
            and abs(position[1]) <= self.TABLE_ABS_Y_BOUND
        )

        if current_contact and lifted and object_in_bounds:
            if self.lift_started_wall_time is None:
                self.lift_started_wall_time = wall_time
            elif wall_time - self.lift_started_wall_time >= self.hold_time:
                self.success = True
        else:
            self.lift_started_wall_time = None

        wrist_position = self.mj_data.xpos[
            self.mj_model.body("right_wrist_yaw_link").id
        ]
        _TRACE.emit(
            "sim_state",
            task_time_s=wall_time - self.armed_wall_time,
            sim_time_s=self.simulator_elapsed,
            object_z=object_z,
            wrist_x=float(wrist_position[0]),
            wrist_y=float(wrist_position[1]),
            wrist_z=float(wrist_position[2]),
            hand_closure=hand_closure,
            contact=current_contact,
            lifted=lifted,
        )

        if self.scenario == "bottle_apple":
            other = "apple" if self.target == "bottle" else "bottle"
            other_body = self.mj_model.body(f"{other}_body")
            other_initial_z = 0.842 if other == "apple" else 0.875
            self.wrong_object_lifted = self.wrong_object_lifted or (
                float(self.mj_data.xpos[other_body.id][2])
                >= other_initial_z + self.lift_height
            )

        with self.reward_lock:
            self.last_reward = float(self.success)
        self._write_metrics()

    def check_fall(self):
        if self.lock_lower_body and self._locked_root_qpos is not None:
            if self.mj_data.qpos[2] < 0.2:
                self.robot_falls += 1
            self.fall = False
            self._apply_lower_body_lock()
            return
        super().check_fall()
        if self.fall:
            self._last_metrics_write = -1.0

    def _write_metrics(self, force: bool = False):
        if not self.metrics_path:
            return
        wall_time = time.monotonic()
        task_time = (
            None if self.armed_wall_time is None else wall_time - self.armed_wall_time
        )
        simulator_time = None if self.armed_at is None else self.simulator_elapsed
        if not force and wall_time - self._last_metrics_write < 0.25:
            return
        self._last_metrics_write = wall_time
        position = (
            self.mj_data.xpos[self.object_body.id].tolist()
            if hasattr(self, "object_body")
            else None
        )
        object_quaternion = (
            self.mj_data.xquat[self.object_body.id].tolist()
            if hasattr(self, "object_body")
            else None
        )
        right_hand_positions = {
            name: self.mj_data.xpos[self.mj_model.body(name).id].tolist()
            for name in (
                "right_wrist_yaw_link",
                "right_hand_thumb_2_link",
                "right_hand_middle_1_link",
                "right_hand_index_1_link",
            )
        }
        right_hand_joint_positions = {
            name: float(
                self.mj_data.qpos[
                    self.mj_model.jnt_qposadr[self.mj_model.joint(name).id]
                ]
            )
            for name in (
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint",
                "right_hand_index_0_joint",
                "right_hand_index_1_joint",
            )
        }
        object_off_table = bool(
            task_time is not None
            and position is not None
            and (
                position[2] < 0.2
                or not self.TABLE_X_BOUNDS[0] <= position[0] <= self.TABLE_X_BOUNDS[1]
                or abs(position[1]) > self.TABLE_ABS_Y_BOUND
            )
        )
        valid_success = bool(
            getattr(self, "success", False)
            and not object_off_table
            and not self.simulator_unstable
        )
        if self.armed_at is None:
            status = "waiting"
        elif self.simulator_unstable:
            status = "simulator_unstable"
        elif object_off_table:
            status = "object_off_table"
        elif valid_success:
            status = "success"
        elif task_time >= self.task_duration:
            status = "complete"
        else:
            status = "running"
        if status in {"success", "object_off_table", "simulator_unstable", "complete"}:
            if status != self._trace_terminal_status:
                _TRACE.emit(
                    "task_success" if status == "success" else "task_failed",
                    status=status,
                    task_time_s=task_time,
                    simulator_time_s=simulator_time,
                    object_z=None if position is None else float(position[2]),
                )
                self._trace_terminal_status = status
        payload = {
            "schema_version": 1,
            "scenario": self.scenario,
            "target": self.target,
            "seed": self.seed,
            "status": status,
            "task_time_s": task_time,
            "simulator_time_s": simulator_time,
            "task_duration_s": self.task_duration,
            "object_position": position,
            "object_quaternion": object_quaternion,
            "right_hand_positions": right_hand_positions,
            "right_hand_joint_positions": right_hand_joint_positions,
            "initial_object_z": getattr(self, "initial_object_z", None),
            "max_object_z": getattr(self, "max_object_z", None),
            "contact_observed": getattr(self, "contact_observed", False),
            "right_hand_contact": getattr(self, "current_contact", False),
            "lift_observed": bool(
                hasattr(self, "max_object_z")
                and self.max_object_z >= self.initial_object_z + self.lift_height
            ),
            "success": valid_success,
            "object_off_table": object_off_table,
            "wrong_object_lifted": getattr(self, "wrong_object_lifted", False),
            "robot_falls": getattr(self, "robot_falls", 0),
            "simulator_instabilities": self.simulator_instabilities,
            "lower_body_locked": self.lock_lower_body,
            "waist_yaw_bounds_rad": (
                None
                if self.waist_yaw_bounds is None
                else self.waist_yaw_bounds.tolist()
            ),
            "waist_yaw_position_rad": float(
                self.mj_data.qpos[
                    self.mj_model.jnt_qposadr[
                        self.mj_model.joint("waist_yaw_joint").id
                    ]
                ]
            ),
            "assisted_grasp_enabled": self.assisted_grasp_enabled,
            "assisted_grasp_active": self.assisted_grasp_active,
            "assisted_grasp_activations": self.assisted_grasp_activations,
            "assisted_grasp_contact_grace_s": self.assisted_grasp_contact_grace,
            "reset_applied": self.reset_applied,
            "criteria": {
                "lift_height_m": self.lift_height,
                "hold_time_s": self.hold_time,
                "requires_right_hand_contact": True,
            },
        }
        path = Path(self.metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
        if self.metrics_history_path:
            history_path = Path(self.metrics_history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("a") as history:
                history.write(json.dumps(payload, sort_keys=True) + "\n")

    def get_privileged_obs(self):
        return {
            f"{self.target}_pos": self.mj_data.xpos[self.object_body.id].copy(),
            f"{self.target}_quat": self.mj_data.xquat[self.object_body.id].copy(),
        }


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        elif env_name == "pnp_bottle":
            self.sim_env = BottleTaskEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Supported environments are 'default' and 'pnp_bottle'."
            )

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
