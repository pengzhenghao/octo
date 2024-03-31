from pathlib import Path

import numpy as np
import tqdm
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.obs.image_obs import ImageObservation

from pvp.sb3.common.save_util import load_from_zip_file
from pvp.sb3.common.save_util import save_to_pkl
from pvp.sb3.common.vec_env import DummyVecEnv
from pvp.sb3.ppo import PPO
from pvp.sb3.ppo.policies import ActorCriticPolicy

if __name__ == '__main__':

    REPO_ROOT = Path(__file__).parent.parent.resolve()
    DATASET_ROOT = Path(__file__).parent.resolve()

    N_TRAIN_EPISODES = 200
    N_VAL_EPISODES = 200
    # EPISODE_LENGTH = 10

    # Initialize environment
    train_env_config = dict(
        use_render=False,  # Open the interface
        manual_control=False,  # Allow receiving control signal from external device
        # controller=control_device,
        window_size=(200, 200),
        horizon=1500,

        # num_scenarios=1,

        # Also add some rendering!!!!

    )

    train_env_config.update(dict(
        use_render=True,
        image_observation=False,
        norm_pixel=False,
        sensors=dict(rgb_camera=(RGBCamera, 512, 256)),
        # render_pipeline=True
    ))


    def _make_train_env():
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        # from pvp.sb3.common.monitor import Monitor
        train_env = HumanInTheLoopEnv(config=train_env_config)
        # train_env = Monitor(env=train_env, filename=str(trial_dir))
        return train_env


    n_envs = 1
    vec_env_cls = DummyVecEnv
    env = _make_train_env()

    # Initialize agent
    algo_config = dict(
        policy=ActorCriticPolicy,
        n_steps=1024,  # n_steps * n_envs = total_batch_size
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        # tensorboard_log=trial_dir,
        create_eval_env=False,
        verbose=2,
        # seed=seed,
        device="auto",
        env=env
    )
    model = PPO(**algo_config)

    ckpt = REPO_ROOT / "metadrive_pvp_20m_steps"

    print(f"Loading checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(
        ckpt, device=model.device, print_system_info=False
    )
    model.set_parameters(params, exact_match=True, device=model.device)
    print(f"Model is loaded from {ckpt}!")


    def create_fake_episode(path, env, ep_count):
        episode = []

        # cfg = env.config.copy()
        # cfg["stack_size"] = 1

        # img_obs = ImageObservation(cfg, image_source="rgb_camera", clip_rgb=False)

        observations = env.reset()
        env.engine.force_fps.disable()

        states = None
        deterministic = False
        step = 0
        while True:
            actions, states = model.predict(
                observations, state=states, episode_start=None, deterministic=deterministic
            )

            image = env.engine.get_sensor("rgb_camera").perceive(to_float=False)
            image = image[..., [2, 1, 0]]

            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # plt.show()

            new_observations, rewards, dones, infos = env.step(actions)

            if infos["navigation_forward"]:
                language_instruction = "Move forward"
            elif infos["navigation_left"]:
                language_instruction = "Turn left"
            elif infos["navigation_right"]:
                language_instruction = "Turn right"
            else:
                raise ValueError("Something wrong in the navigation command: {}".format(infos["navigation_command"]))

            episode.append({
                'image': np.asarray(image, dtype=np.uint8),
                # 'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
                'state': observations,
                'action': actions,
                'language_instruction': language_instruction,
            })
            observations = new_observations

            if dones:
                break

            step += 1

        path = Path(path) / "episode_{}.pkl".format(ep_count)
        path = path.resolve()
        save_to_pkl(path=path, obj=episode)
        print("File saved at: ", path)

        # return 0, 0, 0
        return float(infos["arrive_dest"]), float(infos["episode_reward"]), float(infos["episode_length"])


    # create fake episodes for train and validation
    print("Generating train examples...")
    # os.makedirs('data/train', exist_ok=True)
    succl, rel, lel = [], [], []
    for ep_count in tqdm.tqdm(range(N_TRAIN_EPISODES)):
        succ, rew, le = create_fake_episode(DATASET_ROOT / f'data/train/', env=env, ep_count=ep_count)
        succl.append(succ)
        rel.append(rew)
        lel.append(le)
    print(f"TRAIN Success {np.mean(succl)}, Reward {np.mean(rel)}, Length {np.mean(lel)}")
    env.close()

    train_env_config["start_seed"] = 1000


    def _make_eval_env():
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        eval_env = HumanInTheLoopEnv(config=train_env_config)
        # eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env


    env = _make_eval_env()
    succl, rel, lel = [], [], []
    for ep_count in tqdm.tqdm(range(N_VAL_EPISODES)):
        succ, rew, le = create_fake_episode(DATASET_ROOT / f'data/eval/', env=env, ep_count=ep_count)
        succl.append(succ)
        rel.append(rew)
        lel.append(le)
    print(f"EVAL Success {np.mean(succl)}, Reward {np.mean(rel)}, Length {np.mean(lel)}")
    env.close()
