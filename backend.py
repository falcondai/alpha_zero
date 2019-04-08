import tensorflow as tf

from base import SharedStorage, ReplayBuffer, AlphaZeroConfig, Network


def train_network(config: AlphaZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    network = Network()
    optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule, config.momentum)
    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch()
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
    loss = 0
    for image, (target_value, target_policy) in batch:
        value, policy_logits = network.inference(image)
        loss += (
            tf.losses.mean_squared_error(value, target_value) +
            tf.nn.softmax_cross_entropy_with_logits(
                logits=policy_logits, labels=target_policy))

    for weights in network.get_weights():
        loss += weight_decay * tf.nn.l2_loss(weights)

    optimizer.minimize(loss)


def launch_job(f, *args):
    f(*args)
