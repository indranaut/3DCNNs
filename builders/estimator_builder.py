import os
import tensorflow as tf


def build_estimator_config(proto_config):
    experiment_dir = proto_config.experiment_dir
    experiment_name = proto_config.experiment_name
    experiment_dir = os.path.join(experiment_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    model_dir = proto_config.model_dir_name
    model_dir = os.path.join(experiment_dir, model_dir)
    os.makedirs(model_dir, exist_ok=True)
    tf_random_seed = proto_config.tf_random_seed
    save_summary_steps = proto_config.save_summary_steps
    keep_checkpoint_max = proto_config.keep_checkpoint_max
    log_step_count_steps = proto_config.log_step_count_steps
    save_checkpoints_steps = proto_config.save_checkpoints_steps
    use_xla = proto_config.use_xla
    if use_xla:
        tf.config.optimizer.set_jit(True)

    estimator_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        tf_random_seed=tf_random_seed,
        save_summary_steps=save_summary_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=keep_checkpoint_max,
        log_step_count_steps=log_step_count_steps,
    )
    return estimator_config
