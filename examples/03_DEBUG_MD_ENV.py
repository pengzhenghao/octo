import sys
import pathlib
import tqdm
from absl import app, flags, logging
import gym
import jax
import numpy as np
import wandb

# sys.path.append("path/to/your/act")

# from envs.aloha_sim_env import AlohaGymEnv  # keep this to register ALOHA sim env

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper, UnnormalizeActionProprioAction, UnnormalizeActionProprio, ResizeImageWrapper
from gym import ObservationWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", "/data/zhenghao/octo/models/finetune_metadrive_2024-02-22_2346", "Path to finetuned Octo checkpoint directory."
)

flags.DEFINE_bool(
    "wandb",
    False,
    "Whether wandb",
)

flags.DEFINE_integer(
    "step",
    None,
    "Step",
)


flags.DEFINE_integer(
    "num_rollouts",
    20,
    "Number of rollouts to run.",
)

# flags.DEFINE_string(
#     "exp_name",
#     "eval_metadrive",
#     "name",
# )

class MetaDriveObsWrapper(ObservationWrapper):
    observation_space = gym.spaces.Dict(
        {"image_primary": gym.spaces.Box(shape=(256, 512, 3), dtype=np.uint8, low=0, high=255),

         # "state": gym.spaces.Box(shape=(259,), dtype=np.float32, low=float("-inf"), high=float("+inf"))
         "state": gym.spaces.Box(shape=(19,), dtype=np.float32, low=float("-inf"), high=float("+inf"))

         }
    )

    def observation(self, observation):
        cam = self.env.engine.get_sensor("rgb_camera")
        image = cam.perceive(to_float=False)
        image = image[..., [2, 1, 0]]


        # TODO: hardcoded here to remove "lidar".
        observation = observation[..., :19]

        return {"image_primary": image, "proprio":  observation}




def main(_):

    # setup wandb for logging
    if not FLAGS.wandb:
        import os
        os.environ["WANDB_MODE"] = "offline"
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

    # from jax.config import config
    #
    # config.update('jax_disable_jit', True)

    num_rollouts = FLAGS.num_rollouts

    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")

    # setup wandb for logging

    finetuned_path = pathlib.Path(FLAGS.finetuned_path)
    # wandb_id = finetuned_path.name
    # exp_name = FLAGS.exp_name
    exp_name = finetuned_path.name
    group = f"{exp_name}_EVAL"
    wandb_id = f"{exp_name}_EVAL_step{FLAGS.step}"

    print(f"{wandb_id=}, {group=}, {exp_name=}")


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
    env_config = dict(
        manual_control=False,  # Allow receiving control signal from external device
        # controller=control_device,
        window_size=(200, 200),
        horizon=1500,
        use_render=True,
        image_observation=False,
        norm_pixel=False,
        sensors=dict(rgb_camera=(RGBCamera, 512, 256)),
        # render_pipeline=True

        start_seed=1000,
    )


    def _make_env():
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv5TupleReturn
        train_env = HumanInTheLoopEnv5TupleReturn(config=env_config)
        return train_env


    env = _make_env()

    env = MetaDriveObsWrapper(env)


    # add wrappers for history and "receding horizon control", i.e. action chunking
    env = HistoryWrapper(env, horizon=5)
    env = RHCWrapper(env, exec_horizon=5)
    env = ResizeImageWrapper(env, resize_size=(256, 256))

    # TEST
    # ret = env.reset()

    # load finetuned model
    logging.info("Loading finetuned model...")



    # images_list = []
    succ_list = []
    ep_reward_list = []

    # running rollouts
    for ep_count in range(num_rollouts):
        obs, info = env.reset()

        # run rollout for 400 steps
        images = [o for o in obs["image_primary"]]

        episode_return = 0.0

        action_list = []

        # while len(images) < 100:
        horizon = 1000
        for step_count in tqdm.trange(horizon,):

            if isinstance(info["navigation_forward"], bool):
                navigation_forward = info["navigation_forward"]
                navigation_left = info["navigation_left"]
                navigation_right = info["navigation_right"]
            else:
                navigation_forward = info["navigation_forward"][-1]
                navigation_left = info["navigation_left"][-1]
                navigation_right = info["navigation_right"][-1]
            if navigation_forward:
                language_instruction = "Move forward"
            elif navigation_left:
                language_instruction = "Turn left"
            elif navigation_right:
                language_instruction = "Turn right"
            else:
                raise ValueError("Something wrong in the navigation command: {}".format(info["navigation_command"]))
            print("Instruction: ", language_instruction, info['navigation_command'])

            # create task specification --> use model utility to create task dict with correct entries

            # model returns actions of shape [batch, pred_horizon, action_dim] -- remove batch
            # actions = model.sample_actions(
            #     jax.tree_map(lambda x: x[None], obs), task, rng=jax.random.PRNGKey(0)
            # )

            # PZH: Debug:
            actions = np.array([[[0, 1]] * 5])

            actions = actions[0]

            # step env -- info contains full "chunk" of observations for logging
            # obs only contains observation for final step of chunk
            obs, reward, done, trunc, info = env.step(actions)

            action_list.append(actions)

            # PZH
            images.extend([o for o in obs["image_primary"]])
            # images.extend([o["image_primary"][0] for o in info["observations"]])

            episode_return += reward
            if done or trunc:
                break
        print(f"Episode return: {episode_return}")
        print(f"Success: {info['arrive_dest'][-1]}")

        ep_reward_list.append(episode_return)
        succ_list.append(info['arrive_dest'][-1])

    #     wandb.log(
    #         {"rollout_video": wandb.Video(np.array(images).transpose(0, 3, 1, 2),
    #                                       caption=f"Succ:{info['arrive_dest'][-1]},Rew{episode_return:.1f}", fps=20),
    #          "actions_0": wandb.Histogram(np.array(action_list)[..., 0].flatten()),
    #          "actions_1": wandb.Histogram(np.array(action_list)[..., 1].flatten()),
    #          "episode_return": episode_return,
    #          "success": info['arrive_dest'][-1],
    #          },
    #         step=FLAGS.step + ep_count
    #     )
    # wandb.log({"avg_episode_return": np.mean(ep_reward_list), "avg_success": np.mean(succ_list)}, step=FLAGS.step + ep_count)
    # print(f"Episode return: {np.mean(ep_reward_list)}, Success: {np.mean(succ_list)}")

if __name__ == "__main__":
    app.run(main)