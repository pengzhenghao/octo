nohup python examples/02_finetune_new_observation_action_METADRIVE.py \
--config=examples/configs/finetune_metadrive_config.py:full,language_conditioned \
--name=0401_Full_NoLidar_Lang \
> 0401.log 2>&1 &