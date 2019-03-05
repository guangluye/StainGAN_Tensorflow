import tensorflow as tf
import random

def cycle_consistency_loss(x, cyc_x, y, cyc_y, lambd=10):
    return lambd * (tf.reduce_mean(tf.abs(x - cyc_x)) + tf.reduce_mean(tf.abs(y - cyc_y)))


def generator_loss(fake_rec, REAL_LABEL=1):
    return tf.reduce_mean(tf.squared_difference(fake_rec, REAL_LABEL))


def discriminator_loss(real_rec, fake_rec, REAL_LABEL=1):
    return (tf.reduce_mean(tf.squared_difference(real_rec, REAL_LABEL)) + tf.reduce_mean(tf.square(fake_rec))) / 2.0


def make_optimizer(loss, variables, start_learning_rate, beta1=0.9, beta2=0.999, name='Adam'):
    global_step = tf.Variable(0, trainable=False)
    end_learning_rate = 0.000001
    decay_steps = 10000
    learning_rate = tf.train.polynomial_decay(start_learning_rate, global_step, decay_steps, end_learning_rate, power=1.0, cycle=True)

    learning_step = (
        tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
            .minimize(loss, global_step=global_step, var_list=variables)
    )
    return learning_step


def fake_image_pool(num_fakes, fake, fake_pool):
    """使用之前生成器产生的图片来训练判别器，而不是使用最新的生成器产生的图片，这样能够减小model oscillation"""

    if (num_fakes < 50):
        fake_pool[num_fakes] = fake
        return fake
    else:
        p = random.random()
        if p > 0.5:
            random_id = random.randint(0, 50 - 1)
            temp = fake_pool[random_id]
            fake_pool[random_id] = fake
            return temp
        else:
            return fake
