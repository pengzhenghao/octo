nohup python examples/02_finetune_new_observation_action_METADRIVE.py \
--config=examples/configs/finetune_metadrive_config.py:full,language_conditioned \
--config.batch_size=32 \
--name=0401_Full_NoLidar_Lang_NoNavi \
> 0401_NoNavi.log 2>&1 &