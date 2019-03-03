import tensorflow as tf
from Generator import Generator
from Discriminator import Discriminator
from build_dataset import build_dataset
from utils import cycle_consistency_loss, generator_loss, discriminator_loss, make_optimizer, convert2int
import os


os.environ["CUDA_VISIBLE_DEVICES"]="0"
tfrecords_Xpath = './data/trainX_tfrecords'
tfrecords_Ypath = './data/trainY_tfrecords'
image_width = 400
image_height = 300
batch_size = 2
num_epochs = 100
learning_rate = 0.001
beta1 = 0.5
shuffle_buffer = 100  #定义随机打乱数据时buffer的大小
ngf = 32
ndf = 64
MODEL_SAVE_PATH = './checkpoints'
MODEL_NAME = 'model.ckpt'

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

data_X = build_dataset(tfrecords_Xpath, image_width, image_height, batch_size, num_epochs, shuffle_buffer)
iterator_X = data_X.make_initializable_iterator()
image_batch_X = iterator_X.get_next()

data_Y = build_dataset(tfrecords_Ypath, image_width, image_height, batch_size, num_epochs, shuffle_buffer)
iterator_Y = data_Y.make_initializable_iterator()
image_batch_Y = iterator_Y.get_next()

G = Generator(ngf, 'G')
F = Generator(ngf, 'F')
D_X = Discriminator(ndf, 'D_X')
D_Y = Discriminator(ndf, 'D_Y')

real_X = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 3])
real_Y = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 3])
fake_X = F(real_Y)
fake_Y = G(real_X)
cyc_X = F(fake_Y)
cyc_Y = G(fake_X)
rec_X = D_X(real_X)
rec_Y = D_Y(real_Y)
fake_rec_X = D_X(fake_X)
fake_rec_Y = D_Y(fake_Y)

cycle_loss = cycle_consistency_loss(real_X, cyc_X, real_Y, cyc_Y)

G_gen_loss = generator_loss(fake_rec_Y)
G_loss = G_gen_loss + cycle_loss
F_gen_loss = generator_loss(fake_rec_X)
F_loss = F_gen_loss + cycle_loss

D_X_loss = discriminator_loss(rec_X, fake_rec_X)
D_Y_loss = discriminator_loss(rec_Y, fake_rec_Y)


# summary
tf.summary.histogram('D_Y/true', rec_Y)
tf.summary.histogram('D_Y/fake', fake_rec_Y)
tf.summary.histogram('D_X/true', rec_X)
tf.summary.histogram('D_X/fake', fake_rec_X)

tf.summary.scalar('loss/G', G_loss)
tf.summary.scalar('loss/D_Y', D_Y_loss)
tf.summary.scalar('loss/F', F_loss)
tf.summary.scalar('loss/D_X', D_X_loss)
tf.summary.scalar('loss/cycle', cycle_loss)

tf.summary.image('X', real_X)
tf.summary.image('Y', real_Y)
tf.summary.image('X/generated', fake_Y)
tf.summary.image('X/reconstruction', cyc_X)
tf.summary.image('Y/generated', fake_X)
tf.summary.image('Y/reconstruction', cyc_Y)

merged = tf.summary.merge_all()

model_vars = tf.trainable_variables()

D_X_vars = [var for var in model_vars if 'D_X' in var.name]
G_vars = [var for var in model_vars if 'G' in var.name]
D_Y_vars = [var for var in model_vars if 'D_Y' in var.name]
F_vars = [var for var in model_vars if 'F' in var.name]

G_optimizer = make_optimizer(G_loss, G_vars, learning_rate, beta1, name='Adam_G')
D_Y_optimizer = make_optimizer(D_Y_loss, D_Y_vars, learning_rate, beta1, name='Adam_D_Y')
F_optimizer = make_optimizer(F_loss, F_vars, learning_rate, beta1, name='Adam_F')
D_X_optimizer = make_optimizer(D_X_loss, D_X_vars, learning_rate, beta1, name='Adam_D_X')

# with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
#     optimizer = tf.no_op(name='optimizers')

saver = tf.train.Saver()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)
    step = 0
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    sess.run(iterator_X.initializer)
    sess.run(iterator_Y.initializer)
    while True:
        try:

            X, Y = sess.run([image_batch_X, image_batch_Y])

            _, D_Y_loss_val = sess.run([D_Y_optimizer, D_Y_loss], feed_dict={real_X: X, real_Y: Y})
            for i in range(0, 5):
                _, G_loss_val = sess.run([G_optimizer, G_loss], feed_dict={real_X: X, real_Y: Y})

            _, D_X_loss_val = sess.run([D_X_optimizer, D_X_loss], feed_dict={real_X: X, real_Y: Y})
            for i in range(0, 5):
                _, F_loss_val = sess.run([F_optimizer, F_loss], feed_dict={real_X: X, real_Y: Y})

            summary = sess.run(merged, feed_dict={real_X: X, real_Y: Y})
            summary_writer.add_summary(summary, step)
            if step % 100 == 0:
                print('-----------Step %d:-------------' % step)
                # print('  Cycle_loss   : {}'.format(Cycle_loss_val))
                print('  G_loss   : {}'.format(G_loss_val))
                print('  D_Y_loss : {}'.format(D_Y_loss_val))
                print('  F_loss   : {}'.format(F_loss_val))
                print('  D_X_loss : {}'.format(D_X_loss_val))
            if step % 100 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)
            step += 1
        except tf.errors.OutOfRangeError:
            print("Training Progress Done!")
            break
    summary_writer.close()


