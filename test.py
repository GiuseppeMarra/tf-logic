from tensorflow.contrib.eager.python import tfe
eager = True
if eager: tfe.enable_eager_execution()
import unittest
import tfl
import tensorflow as tf
import numpy as np
from tfl import World

class TestParser(unittest.TestCase):

    def test_cache(self):

        World.reset()


        a = tf.zeros([10,10])

        print(a)

        """Checking caching mechanism works"""

        # Program Model
        nn1 = lambda x: tf.constant([0., 1, 0])
        nn2 = lambda x: tf.constant([0., 0, 0])
        images = tfl.Domain(label="Images", data=[[0., 0], [1, 1], [0.2, 0.3]])
        zero = tfl.Predicate(label="zero", domains=["Images"], function=nn1)
        one = tfl.Predicate(label="one", domains=["Images"], function=nn2)
        close = tfl.Predicate(label="close", domains=["Images", "Images"],
                              function=lambda x,y: tf.reduce_sum(tf.abs(x - y), axis=1))
        tfl.setTNorm(id=tfl.PRODUCT, p=None)

        # Constraint 1
        "zero(x) and one(y)"
        x = tfl.variable(images)
        y = tfl.variable(images)
        a = tfl.atom(zero, (x,))
        b = tfl.atom(one, (y,))
        andd = tfl.and_n(a, b)

        # Constraint 2
        z = tfl.variable(images)
        h = tfl.variable(images)
        c = tfl.atom(zero, (z,))
        d = tfl.atom(one, (h,))
        papapa = tfl.and_n(c, d)
        ab = tfl.atom(close, (x, y))

        assert len(World._predicates_cache) == 3  # one of zero, one for one and one for close



    def test_inside(self):
        """This test is for evaluating caching when the same variable is used as two different arguments of the same predicate"""

        World.reset()


        def inside(x, y):
            centers_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(x[:, 0:2], y[:, 0:2]), axis=1) + 1e-6)
            return tf.cast((centers_distance + x[:, 2]) < y[:, 2], tf.float32)

        circles = tfl.Domain(label="Circles", data=[[0., 0,  1], [0,0, 2], [0,0, 3]])
        inside = tfl.Predicate(label="inside", domains=["Circles", "Circles"], function=inside)
        tfl.setTNorm(id=tfl.SS, p=1)
        sess = tf.Session()


        # Constraint 1
        "zero(x) and one(y)"
        x = tfl.variable(circles, name="x")
        y = tfl.variable(circles, name="y")
        z = tfl.variable(circles, name="z")
        a = tfl.atom(inside, (x,y))
        b = tfl.atom(inside, (y,z))
        c = tfl.atom(inside, (x,z))
        andd = tfl.and_n(a, b)

        rule = tfl.implies(andd,c)

        assert np.equal(sess.run(rule), np.zeros(shape=[3,3,3])).all()
        assert len(World._predicates_cache)==1

    def test_transposition(self):
        """This test is for evaluating caching when the same variable is used as two different arguments of the same predicate"""

        World.reset()


        def inside(x, y):
            centers_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(x[:, 0:2], y[:, 0:2]), axis=1) + 1e-6)
            return tf.cast((centers_distance + x[:, 2]) < y[:, 2], tf.float32)

        circles = tfl.Domain(label="Circles", data=[[0., 0,  1], [0,0, 2], [0,0, 3]])
        inside = tfl.Predicate(label="inside", domains=["Circles", "Circles"], function=inside)
        tfl.setTNorm(id=tfl.SS, p=1)
        sess = tf.Session()


        # Constraint 1
        x = tfl.variable(circles, name="x")
        y = tfl.variable(circles, name="y")
        a = tfl.atom(inside, (x,y))
        b = tfl.atom(inside, (y,x))
        rule = tfl.and_n(a, b)

        assert np.greater(sess.run(rule), np.zeros(shape=[3,3,3])).all()
        assert len(World._predicates_cache)==1

    def test_logic_mode(self):
        """This test is for evaluating caching when the same variable is used as two different arguments of the same predicate"""

        World.reset()
        World._evaluation_mode = tfl.LOGIC_MODE

        def inside(x, y):
            centers_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(x[:, 0:2], y[:, 0:2]), axis=1) + 1e-6)
            return tf.cast((centers_distance + x[:, 2]) < y[:, 2], tf.float32)

        circles = tfl.Domain(label="Circles", data=[[0., 0, 1], [0, 0, 2], [0, 0, 3]])
        inside = tfl.Predicate(label="inside", domains=["Circles", "Circles"], function=inside)
        tfl.setTNorm(id=tfl.SS, p=1)
        sess = tf.Session()

        # Constraint 1
        x = tfl.variable(circles, name="x")
        y = tfl.variable(circles, name="y")
        a = tfl.atom(inside, (x, y))
        b = tfl.atom(inside, (y, x))
        rule = tfl.and_n(a, b)

        assert np.equal(sess.run(rule), np.zeros(shape=[3, 3, 3])).all()
        assert len(World._predicates_cache) == 1

    def test_callable_atoms(self):
        """This test is for evaluating caching when the same variable is used as two different arguments of the same predicate"""

        World.reset()
        World._evaluation_mode = tfl.LOGIC_MODE

        def inside_impl(x, y):
            centers_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(x[:, 0:2], y[:, 0:2]), axis=1) + 1e-6)
            return tf.cast((centers_distance + x[:, 2]) < y[:, 2], tf.float32)

        circles = tfl.Domain(label="Circles", data=[[0., 0, 1], [0, 0, 2], [0, 0, 3]])
        inside = tfl.Predicate(label="inside", domains=["Circles", "Circles"], function=inside_impl)
        tfl.setTNorm(id=tfl.SS, p=1)
        sess = tf.Session()

        # Constraint 1
        x = tfl.variable(circles, name="x")
        y = tfl.variable(circles, name="y")
        a = inside(x, y)
        b = inside(y, x)
        rule = tfl.and_n(a, b)

        assert np.equal(sess.run(rule), np.zeros(shape=[3, 3, 3])).all()
        assert len(World._predicates_cache) == 1

    def test_parser_atom(self):

        World.reset()
        World._evaluation_mode = tfl.LOGIC_MODE
        def inside(x, y):
            centers_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(x[:, 0:2], y[:, 0:2]), axis=1) + 1e-6)
            return tf.cast((centers_distance + x[:, 2]) < y[:, 2], tf.float32)
        tfl.setTNorm(id=tfl.SS, p=1)
        circles = tfl.Domain(label="Circles", data=[[0., 0, 1], [0, 0, 2], [0, 0, 3]])
        inside = tfl.Predicate(label="inside", domains=["Circles", "Circles"], function=inside)


        x = tfl.variable(circles, "x")
        y = tfl.variable(circles, "y")
        a = tfl.atom(inside, (x,y))

        tensor = tfl.constraint("inside(x,y)")

        sess = tf.Session()
        assert np.equal(sess.run(tensor),sess.run(a)).all()

    def test_parser_and(self):

        World.reset()
        World._evaluation_mode = tfl.LOGIC_MODE
        def inside(x, y):
            centers_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(x[:, 0:2], y[:, 0:2]), axis=1) + 1e-6)
            return tf.cast((centers_distance + x[:, 2]) < y[:, 2], tf.float32)

        tfl.setTNorm(id=tfl.SS, p=1)
        circles = tfl.Domain(label="Circles", data=[[0., 0, 1], [0, 0, 2], [0, 0, 3]])
        inside = tfl.Predicate(label="inside", domains=["Circles", "Circles"], function=inside)


        x = tfl.variable(circles, "x")
        y = tfl.variable(circles, "y")
        a = tfl.atom(inside, (x,y))
        b = tfl.atom(inside, (y,x))
        f = tfl.and_n(a,b)

        tensor = tfl.constraint("inside(x,y) and inside(y,x)")

        sess = tf.Session()


        assert np.equal(sess.run(tensor),sess.run(f)).all()


    def test_parser_forall(self):

        World.reset()
        World._evaluation_mode = tfl.LOGIC_MODE
        tfl.setTNorm(id=tfl.SS, p=1)

        sess = tf.Session()

        def inside(x, y):
            centers_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(x[:, 0:2], y[:, 0:2]), axis=1) + 1e-6)
            return tf.cast((centers_distance + x[:, 2]) < y[:, 2], tf.float32)

        circles = tfl.Domain(label="Circles", data=[[0., 0, 1], [0, 0, 2], [0, 0, 3]])
        inside = tfl.Predicate(label="inside", domains=["Circles", "Circles"], function=inside)

        x = tfl.variable(circles, "x")
        y = tfl.variable(circles, "y")

        symmetric_formula = tfl.forall(x, tfl.forall(y, tfl.and_n(inside(x, y),inside(y, x))))
        symmetric_parsed = tfl.constraint("forall x: forall y: inside(x,y) and inside(y,x)")
        assert np.equal(sess.run(symmetric_formula), sess.run(symmetric_parsed))


    def test_slice_caching(self):
        World.reset()

        """Checking caching mechanism works"""

        # Program Model
        nn = tfl.functions.FeedForwardNN(input_shape=[2], output_size=2)
        images = tfl.Domain(label="Images", data=[[0., 0], [1, 1], [0.2, 0.3]])
        zero = tfl.Predicate(label="zero", domains=["Images"], function=tfl.functions.Slice(nn, 0))
        one = tfl.Predicate(label="one", domains=["Images"], function=tfl.functions.Slice(nn, 1))
        close = tfl.Predicate(label="close", domains=["Images", "Images"],
                              function=lambda x: tf.reduce_sum(tf.abs(x[0] - x[1]), axis=1))
        tfl.setTNorm(id=tfl.PRODUCT, p=None)

        # Constraint 1
        c1 = tfl.constraint("forall x: zero(x) and one(x) and one(y)")


    def test_subdomain(self):
        World.reset()

        """Checking caching mechanism works"""

        # Program Model
        nn = tfl.functions.FeedForwardNN(input_shape=[2], output_size=2)
        images = tfl.Domain(label="Images", data=[[0., 0], [1, 1], [0.2, 0.3]])
        supervised_images = tfl.Domain(label="SupImages", data=[[0., 0], [1, 1]], father=images)
        zero = tfl.Predicate(label="zero", domains=["Images"], function=tfl.functions.Slice(nn, 0))
        one = tfl.Predicate(label="one", domains=["Images"], function=tfl.functions.Slice(nn, 1))
        tfl.setTNorm(id=tfl.PRODUCT, p=None)

        # Constraint 1
        c1 = tfl.constraint("zero(x) and one(x)", {"x": supervised_images})

        x = tfl.variable(supervised_images, name="x")
        c2 = tfl.and_n(zero(x), one(x))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        assert len(sess.run(c1)) == supervised_images.size
        assert np.equal(sess.run(c1), sess.run(c2)).all()




# def test_parser_forall2(self):
    #
    #     World.reset()
    #     World._evaluation_mode = tfl.LOGIC_MODE
    #     def inside(x):
    #         centers_distance = tf.sqrt(tf.reduce_sum(tf.squared_difference(x[0][:, 0:2], x[1][:, 0:2]), axis=1) + 1e-6)
    #         return tf.cast((centers_distance + x[0][:, 2]) < x[1][:, 2], tf.float32)
    #
    #     tfl.setTNorm(id=tfl.SS, p=1)
    #     tfl.Domain(label="Circles", data=[[0., 0, 1], [0, 0, 2], [0, 0, 3]])
    #     tfl.Predicate(label="inside", domains=["Circles", "Circles"], function=inside)
    #
    #
    #     symmetric = tfl.constraint("forall x: forall y: inside(x,y) and inside(y,x)")
    #     antisymmetric = tfl.constraint("forall x: forall y: inside(x,y) and not inside(y,x)")
    #
    #     sess = tf.Session()
    #
    #
    #     assert np.equal(sess.run(symmetric), 0)
    #     assert np.equal(sess.run(antisymmetric), 1)

if __name__ == '__main__':
    unittest.main()