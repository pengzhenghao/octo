"""
This script demonstrates how to load and rollout a finetuned Octo model.
We use the Octo model finetuned on ALOHA sim data from the examples/finetune_new_observation_action.py script.

For installing the ALOHA sim environment, clone: https://github.com/tonyzhaozh/act
Then run:
pip3 install opencv-python modern_robotics pyrealsense2 h5py_cache pyquaternion pyyaml rospkg pexpect mujoco==2.3.3 dm_control==1.0.9 einops packaging h5py

Finally, modify the `sys.path.append` statement below to add the ACT repo to your path.
If you are running this on a head-less server, start a virtual display:
    Xvfb :1 -screen 0 1024x768x16 &
    export DISPLAY=:1

To run this script, run:
    cd examples
    python3 03_eval_finetuned.py --filetuned_path=<path_to_finetuned_aloha_checkpoint>
"""
import sys

import tqdm
from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb

# sys.path.append("path/to/your/act")

# from envs.aloha_sim_env import AlohaGymEnv  # keep this to register ALOHA sim env

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper, UnnormalizeActionProprio, ResizeImageWrapper
from gym import ObservationWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", "/data/zhenghao/octo/models/finetune_metadrive_2024-02-22_1352", "Path to finetuned Octo checkpoint directory."
)

flags.DEFINE_bool(
    "wandb",
    False,
    "Whether wandb",
)

flags.DEFINE_string(
    "exp_name",
    "finetune_metadrive",
    "name",
)

class MetaDriveObsWrapper(ObservationWrapper):
    observation_space = gym.spaces.Dict(
        {"image_primary": gym.spaces.Box(shape=(256, 512, 3), dtype=np.uint8, low=0, high=255),
         "state": gym.spaces.Box(shape=(259,), dtype=np.float32, low=float("-inf"), high=float("+inf"))
         }
    )
    def observation(self, observation):
        cam = self.env.engine.get_sensor("rgb_camera")
        image = cam.perceive(to_float=False)
        image = image[..., [2, 1, 0]]
        return {"image_primary": image, "proprio":  observation}




def main(_):

    # setup wandb for logging
    if not FLAGS.wandb:
        import os
        os.environ["WANDB_MODE"] = "offline"

    # setup wandb for logging
    wandb.init(name="eval_metadrive", project="octo")


    # make gym environment
    ##################################################################################################################
    # environment needs to implement standard gym interface + return observations of the following form:
    #   obs = {
    #     "image_0": ...
    #     "image_1": ...
    #   }
    # it should also implement an env.get_task() function that returns a task dict with goal and/or language instruct.
    #   task = {
    #     "language_instruction": "some string"
    #     "goal": {
    #       "image_0": ...
    #       "image_1": ...
    #     }
    #   }
    ##################################################################################################################
    # env = gym.make("aloha-sim-cube-v0")


    # PZH: Make env here
    from metadrive import MetaDriveEnv
    from metadrive.component.sensors.rgb_camera import RGBCamera

    # Initialize environment
    train_env_config = dict(
        manual_control=False,  # Allow receiving control signal from external device
        # controller=control_device,
        window_size=(200, 200),
        horizon=1500,
        use_render=True,
        image_observation=False,
        norm_pixel=False,
        sensors=dict(rgb_camera=(RGBCamera, 512, 256)),
        # render_pipeline=True
    )


    def _make_train_env():
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv5TupleReturn
        train_env = HumanInTheLoopEnv5TupleReturn(config=train_env_config)
        return train_env


    env = _make_train_env()

    env = MetaDriveObsWrapper(env)


    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=5)
    env = RHCWrapper(env, exec_horizon=5)

    # TODO Which order?
    env = ResizeImageWrapper(env, resize_size=(256, 256))

    # TEST
    ret = env.reset()

    # load finetuned model
    logging.info("Loading finetuned model...")

    finetuned_path = FLAGS.finetuned_path
    model = OctoModel.load_pretrained(finetuned_path)


    # wrap env to handle action/proprio normalization -- match normalization type to the one used during finetuning
    env = UnnormalizeActionProprio(
        env, model.dataset_statistics, normalization_type="normal"
    )

    # running rollouts
    for _ in range(5):
        obs, info = env.reset()

        # create task specification --> use model utility to create task dict with correct entries
        # language_instruction = env.get_task()["language_instruction"]
        # task = model.create_tasks(texts=language_instruction)
        task = model.create_tasks(texts=["Drive the ego vehicle to the destination."])

        # run rollout for 400 steps
        images = [obs["image_primary"]]

        episode_return = 0.0


        # while len(images) < 100:
        horizon = 1000
        for _ in tqdm.trange(horizon,):

            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            actions = model.sample_actions(
                jax.tree_map(lambda x: x[None], obs), task, rng=jax.random.PRNGKey(0)
            )
            actions = actions[0]

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)

            # PZH
            images.extend([o for o in obs["image_primary"]])
            # images.extend([o["image_primary"][0] for o in info["observations"]])

            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")
        print(f"Success: {info['arrive_dest'][-1]}")

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        # wandb.log(
        #     {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2)[::2])}
        # )

        # log rollout video to wandb -- subsample temporally 2x for faster logging
        wandb.log(
            {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2), fps=30)}
        )


if __name__ == "__main__":
    app.run(main)
