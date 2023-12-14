import os
from typing import Callable

import json
import os.path
import tempfile
from typing import List, Optional
from gym import error, logger


from gymnasium.wrappers.monitoring import video_recorder #https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/video_recorder.py

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv




        # self.video_recorder = video_recorder.VideoRecorder(
        #     env=self.env, base_path=base_path, metadata={"step_id": self.step_id}
        # ) #Uses the video recorder from https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/video_recorder.py

base_video_recorder = video_recorder.VideoRecorder #SM

class customVideoRecorder(base_video_recorder):  #SM
    """"""
    def __init__(
        self,
        env,
        path: Optional[str] = None,
        metadata: Optional[dict] = None,
        enabled: bool = True,
        base_path: Optional[str] = None): 
        
        base_video_recorder.__init__(
        self,
        env,
        path,
        metadata,
        enabled,
        base_path)


    def save_video(self, i_video):
        """Flush all data to disk and close any open frame encoders."""
        # print("Heloooooooo")

        if not self.enabled or self._closed:
            return

        # # First close the environment
        # self.env.close()

        # Close the encoder
        if len(self.recorded_frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                raise error.DependencyNotInstalled(
                    "MoviePy is not installed, run `pip install moviepy`"
                )

            temp_path = self.path.split("/")
            temp_name_ = temp_path[-1].split("_")
            temp_name = temp_name_[0:3]+[str(i_video)]+temp_name_[3::]
            name = "_".join(temp_name)
            # name = str(i_video)+"_"+temp_path[-1]
            temp ="/".join(temp_path[:-1])
            path = temp+"/"+name

            logger.debug(f"Closing video encoder: path={self.path}")
            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            clip.write_videofile(path)
        else:
            # No frames captured. Set metadata.
            if self.metadata is None:
                self.metadata = {}
            self.metadata["empty"] = True

        self.write_metadata()

        # Stop tracking this for autoclose
        # self._closed = True



class VecVideoRecorder(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered image as mp4 video.
    It requires ffmpeg or avconv to be installed on the machine.

    :param venv:
    :param video_folder: Where to save videos
    :param record_video_trigger: Function that defines when to start recording.
                                        The function takes the current number of step,
                                        and returns whether we should start recording or not.
    :param video_length:  Length of recorded videos
    :param name_prefix: Prefix to the video name
    """

    # video_recorder: video_recorder.VideoRecorder
    video_recorder: customVideoRecorder

    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
        render_fps: int = 30,
        video_name: str = "",
    ):
        VecEnvWrapper.__init__(self, venv)

        self.env = venv
        # Temp variable to retrieve metadata
        temp_env = venv

        # Unwrap to retrieve metadata dict
        # that will be used by gym recorder
        while isinstance(temp_env, VecEnvWrapper):
            temp_env = temp_env.venv

        if isinstance(temp_env, DummyVecEnv) or isinstance(temp_env, SubprocVecEnv):
            metadata = temp_env.get_attr("metadata")[0]
        else:
            metadata = temp_env.metadata

        metadata["render_fps"]= int(render_fps)
        self.env.metadata = metadata
        # print("Metadata = ", self.env.metadata)

        assert self.env.render_mode == "rgb_array", f"The render_mode must be 'rgb_array', not {self.env.render_mode}"

        self.record_video_trigger = record_video_trigger
        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length
        self.video_name = video_name

        self.recording = False
        self.recorded_frames = 0

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        self.start_video_recorder()
        return obs

    def start_video_recorder(self) -> None:
        self.close_video_recorder()

        video_name = self.video_name+f"_frame-rate-{self.env.metadata['render_fps']}-{self.name_prefix}-step-{self.step_id}-to-step-{self.step_id + self.video_length}"
        base_path = os.path.join(self.video_folder, video_name)

        # self.video_recorder = video_recorder.VideoRecorder(
        #     env=self.env, base_path=base_path, metadata={"step_id": self.step_id}
        # ) #Uses the video recorder from https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/video_recorder.py

        self.video_recorder = customVideoRecorder(
            env=self.env, base_path=base_path, metadata={"step_id": self.step_id}
        ) #Uses the video recorder from https://github.com/openai/gym/blob/master/gym/wrappers/monitoring/video_recorder.py



        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.step_id) #returns true when the step_id matched the starting step to record a video

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()

        self.step_id += 1
        if self.recording:
            self.video_recorder.capture_frame()
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                print(f"Saving video to {self.video_recorder.path}")
                self.close_video_recorder()
                

        elif self._video_enabled(): #If start recording
            self.start_video_recorder()

        return obs, rews, dones, infos

    def close_video_recorder(self) -> None:
        if self.recording:
            # self.video_recorder.env.close()
            self.video_recorder.close()
                    # Stop tracking this for autoclose
            # self.video_recorder._closed = True
        self.recording = False
        self.recorded_frames = 1

    def close(self) -> None:
        VecEnvWrapper.close(self)
        self.close_video_recorder()

    def save_video(self, i_video) -> None: #SM
        self.video_recorder.save_video(i_video) #save video without closing the environment

    def __del__(self):
        self.close_video_recorder()
