nohup python examples/02_finetune_new_observation_action_METADRIVE.py \
--config=examples/configs/finetune_metadrive_config.py:full,language_conditioned \
--config.load_pretrained_weights=False \
--name=0401_Full_NoLidar_Lang_NoNavi_NoPretrained \
> 0401_NoNavi_NoPretrained.log 2>&1 &