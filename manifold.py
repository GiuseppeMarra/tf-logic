# from tensorflow.contrib.eager.python import tfe
# eager = True
# if eager: tfe.enable_eager_execution()
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib as mpl
mpl.use('TkAgg')
import tensorflow as tf
from sklearn import datasets as data
import matplotlib.pyplot as plt
import numpy as np
import tfl

np.random.seed(3)
tf.set_random_seed(3)


unsupervised_size = 200
supervised_size = 20
test_size = 200



allX, ally = data.make_circles(n_samples=unsupervised_size+supervised_size+test_size, shuffle=True, noise=0., random_state=None)
X_unsup_np  = allX[0:unsupervised_size]
X_sup_np, y_sup_np = allX[unsupervised_size:unsupervised_size+supervised_size], ally[unsupervised_size:unsupervised_size+supervised_size]
X_sup = tf.cast(X_sup_np, tf.float32)
y_sup = tf.reshape(tf.cast(y_sup_np, tf.float32), [-1 ,1])

X_test_np, y_test_np = allX[unsupervised_size+supervised_size:], ally[unsupervised_size+supervised_size:]
X_test = tf.cast(X_test_np, tf.float32)
y_test = tf.reshape(tf.cast(y_test_np, tf.float32), [-1 ,1])


plt.scatter(X_unsup_np[:,0],X_unsup_np[:,1], color="grey", marker=".")
plt.show()
plt.scatter(X_sup_np[y_sup_np==1,0],X_sup_np[y_sup_np==1,1], color="green", marker="o", label="A")
plt.scatter(X_sup_np[y_sup_np==0,0],X_sup_np[y_sup_np==0,1], color="red", marker="x", label="not A")
plt.legend()
plt.show()
plt.scatter(X_test_np[y_test_np==1,0],X_test_np[y_test_np==1,1], color="green", marker="o", label="A")
plt.scatter(X_test_np[y_test_np==0,0],X_test_np[y_test_np==0,1], color="red", marker="x", label="not A")
plt.legend()
plt.show()



class IsClose(tfl.functions.AbstractFunction):

    def __call__(self, a, b):

        dist = tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1))
        return tf.where(dist<0.12, tf.ones_like(dist), tf.zeros_like(dist))

is_close = IsClose()
is_A = tfl.functions.FeedForwardNN(input_shape=[2], output_size=1, layers = [30])
X_sup_np = X_sup_np.astype(np.float32)
X = np.concatenate((X_sup_np, X_unsup_np), axis=0)
X = X.astype(np.float32)

tfl.World.reset()
tfl.World._evaluation_mode = tfl.LOSS_MODE
tfl.setTNorm(id=tfl.SS, p=1)


Points = tfl.Domain(label="Points", data=X)
SPoints = tfl.Domain(label="SPoints", data=X_sup_np, father=Points)

R1 = tfl.Predicate("A", domains=["Points",], function=is_A)
R2 = tfl.Predicate("isClose", domains=["Points", "Points"], function=is_close)
R3 = tfl.Predicate("SA", domains=["Points",], function= lambda x: tf.squeeze(y_sup))

c_m = tfl.constraint("forall p: forall q: isClose(p,q) -> (A(p) <-> A(q))")
c_s = tfl.constraint("forall p: SA(p) <-> A(p)", {"p": SPoints})



loss_pre_vincoli =  c_s


activate_rules = tf.placeholder(dtype=tf.bool, shape=[])
lr = tf.placeholder(dtype=tf.float32, shape=[])
loss_post_vincoli = loss_pre_vincoli + 0.001*c_m
loss = tf.cond(activate_rules, lambda: loss_post_vincoli, lambda:loss_pre_vincoli)
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

test_outputs = is_A(X_test)[:,0:1]
test_predictions = tf.where(test_outputs > 0.5, tf.ones_like(test_outputs, dtype=tf.float32), tf.zeros_like(test_outputs, dtype=tf.float32))
accuracy = tf.reduce_sum(tf.cast(tf.equal(test_predictions, y_test), tf.float32)) / test_size

sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
sess.run(tf.global_variables_initializer())

epochs = 100000
flag = False
feed_dict = {activate_rules: flag, lr:0.01}

while True:
    _, acc, ll = sess.run((train_op,accuracy, loss), feed_dict)
    if ll<0.01 and flag:
        break
    if ll<0.3:
        xtt, pred = sess.run((X_test, test_predictions))
        pred = np.reshape(pred, [-1])
        plt.scatter(xtt[pred == 0, 0], xtt[pred == 0, 1],color="red", marker="x", label="not A")
        plt.scatter(xtt[pred == 1, 0], xtt[pred == 1, 1], color="green", marker="o", label="A")
        plt.legend()
        plt.show()
        flag = True
        feed_dict = {activate_rules: flag, lr: 0.01}
    print(acc, ll)

X_test, pred = sess.run((X_test,test_predictions))
pred = np.reshape(pred, [-1])
plt.scatter(X_test[pred==0,0],X_test[pred==0,1],color="red", marker="x", label="not A"),
plt.scatter(X_test[pred==1,0],X_test[pred==1,1],color="green", marker="o", label="A"),
plt.legend()
plt.show()



