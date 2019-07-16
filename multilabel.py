from __future__ import print_function
import keras
from tensorflow.contrib.eager.python import tfe
eager = True
if eager: tfe.enable_eager_execution()
import tensorflow as tf
from keras.datasets import cifar10
import tfl
import numpy as np
import os
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"]= "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'



# --------------------- TRAINING PARAMETERS----------------------------------
iterations = 20000
data_augmentation = False
subtract_pixel_mean = True
n = 6
depth = n * 9 + 2
original_num_classes = 10
num_classes = original_num_classes + 7
use_logic = True
transductive = True
minibatch_size = 20
supervided_size = 1000 # -1 means all of them




# -----------------------------DATA------------------------------------------
# Load the CIFAR10 data.
print("Loading data...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, original_num_classes)
y_test = keras.utils.to_categorical(y_test, original_num_classes)




# -----------------------------MODEL------------------------------------------
print("Creating model graph...")
def extended_cifar_activation(x):
    original = tf.nn.softmax(x[:, :original_num_classes])
    onroad_mammal_others = tf.nn.softmax(x[:, original_num_classes:original_num_classes+3])
    flies_not = tf.nn.softmax(x[:, original_num_classes+3:original_num_classes+5])
    animal_transport = tf.nn.softmax(x[:, original_num_classes+5:original_num_classes+7])
    return tf.concat((original, onroad_mammal_others, flies_not, animal_transport), axis=1)


cifar = keras.Sequential()
cifar.add(keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape))
cifar.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
cifar.add(keras.layers.Flatten())
cifar.add(keras.layers.Dense(num_classes))
cifar_log = keras.Model(cifar.input,cifar.output)
cifar.add(keras.layers.Activation(activation=extended_cifar_activation))

cifar = tfl.functions.FromKerasModel(cifar)



#---------------------------------INPUT---------------------------------------------
print("Creating input iterators graph...")

# make a dataset from a numpy array




if transductive:
    dataset_test = tf.data.Dataset.from_tensor_slices(tf.concat((x_train, x_test), axis=0)).batch(minibatch_size // 2).repeat()
    iter_test = dataset_test.make_one_shot_iterator()
    x_trans = iter_test.get_next()


if supervided_size > 0:
    x_train = x_train[:supervided_size]
    y_train = y_train[:supervided_size]


size = minibatch_size if not transductive else minibatch_size//2

dataset_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(size).repeat()
iter_train = dataset_train.make_one_shot_iterator()
x_tr, y_tr = iter_train.get_next()


x_final = x_trans if transductive else x_tr



dataset_eval = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(minibatch_size).repeat()
iter_eval = dataset_eval.make_one_shot_iterator()
x_eval, y_eval = iter_eval.get_next()



# -------------------------------LOGIC------------------------------------------
print("Creating logic component graph...")

predicates_dict = OrderedDict({
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
            "onroad": 10,
            "mammal": 11,
            "otherleaves": 12,
            "flies": 13,
            "notflies": 14,
            "animal": 15,
            "transport": 16
        })


tfl.World.reset()
tfl.World._evaluation_mode = tfl.LOSS_MODE
tfl.setTNorm(id=tfl.SS, p=1)

# Domains Definition
images = tfl.Domain("Images", data=x_final, size=minibatch_size)

# Predicates Definition
for k,v in predicates_dict.items():
    tfl.Predicate(k, domains=("Images",), function=tfl.functions.Slice(cifar, v))


# Constraints
constraints = []
constraints.append(tfl.constraint("forall x: transport(x) <-> airplane(x) or ship(x) or onroad(x)"))
constraints.append(tfl.constraint("forall x: onroad(x) <-> automobile(x) or truck(x)"))
constraints.append(tfl.constraint("forall x: animal(x) <-> bird(x) or frog(x) or mammal(x)"))
constraints.append(tfl.constraint("forall x: mammal(x) <-> cat(x) or deer(x) or dog(x) or horse(x)"))
constraints.append(tfl.constraint("forall x: otherleaves(x) <-> bird(x) or frog(x) or ship(x) or airplane(x)"))
constraints.append(tfl.constraint("forall x: flies(x) <-> bird(x) or airplane(x)"))
constraints.append(tfl.constraint("forall x: notflies(x) <-> cat(x) or dog(x) or horse(x) or deer(x) or truck(x) or ship(x) or automobile(x) or frog(x)"))




# --------------------- LEARNING -----------------------------#
print("Creating learning operations graph...")

y_pred = cifar_log(x_tr)[:,:original_num_classes]
sup_loss = tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y_tr)
constr_loss = tf.add_n(constraints)


contr_weight = tf.placeholder(tf.float32, shape = ())
total_loss = sup_loss + contr_weight*constr_loss
train_op = tf.train.AdamOptimizer(0.001).minimize(total_loss)





# ------------------------------EVALUATION-----------------------------#
print("Creating evaluation operation graph...")

def custom_accuracy(y_true, y_pred):
    y_pred = y_pred[:,:original_num_classes]
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1),
                   tf.argmax(y_pred, axis=-1)),
           tf.float32))

y_pred_eval = cifar(x_eval)
eval_acc = custom_accuracy(y_true=y_eval, y_pred=y_pred_eval)
def evaluate():
    acc = []
    for k in range(len(x_test)//minibatch_size):
        acc.append(sess.run(eval_acc))
    return np.mean(acc)




# ------------------------EXECUTION----------------------------------- #
print("Session initialization the model ...")
sess = tf.Session()
sess.run(tf.global_variables_initializer())


feed_dict = {contr_weight: 0.001}
for i in range(iterations):
    sess.run(train_op, feed_dict=feed_dict)
    if i % 100 == 0:
        print(evaluate())
