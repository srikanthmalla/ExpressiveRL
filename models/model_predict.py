from networks.FCN_one_hidden import *
import tensorflow as tf
import numpy as np
import os
from params import lr
import shutil
class modelPredictor():
    def __init__(self):
        #current observation and action
        self.observation=tf.placeholder(tf.float32,shape=[None,4])
        self.action=tf.placeholder(tf.float32, shape=[None,1])
        self.inputs=tf.concat([self.observation,self.action],axis=1)
        self.next_state=FCN_one_hidden(self.inputs,10,4,'model_predict/')
        self.found_next_state=tf.placeholder(tf.float32, shape=[None,4])

        dir_="./tmp/cart_pole/model_predictor/"
        self.ckpt_dir=dir_+"model.ckpt"
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        #saver
        self.saver = tf.train.Saver()

        #session and initialization
        self.sess=tf.Session()
        # losses
        self.loss= tf.nn.l2_loss(self.found_next_state-self.next_state)
        self.optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
 
        #writers
        tf_logdir="./graphs/model_predict/"
        if os.path.exists(tf_logdir):
            shutil.rmtree(tf_logdir)

        self.writer=tf.summary.FileWriter(tf_logdir,self.sess.graph)

        #initializations
        self.init_op=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init_op)
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()
    def save(self, step):
        self.saver.save(self.sess, self.ckpt_dir)

    def load(self):
        self.saver.restore(self.sess,self.ckpt_dir)

    def predictNextState(self, observation,action):
        return self.sess.run(self.next_state,feed_dict={self.observations:observation,self.action:action})

    def train(self,observation,action,next_observation,step):
        action=np.expand_dims(action,axis=1)
        [_,loss]=self.sess.run([self.optimizer,self.loss],feed_dict={self.observation:observation, self.action:action, self.found_next_state:next_observation})
        lossvalue= tf.Summary(value=[tf.Summary.Value(tag="modelPredictor/loss",simple_value=np.float32(loss))])
        self.writer.add_summary(lossvalue,step)

    def test(self,observation,action,next_observation):
        action=np.expand_dims(action,axis=1)
        [next_state,val_loss]=self.sess.run([self.next_state,self.loss],feed_dict={self.observation:observation,self.action:action,self.found_next_state:next_observation}) 
        print("current:",observation,"next state:",next_observation,"predicted:",next_state,"loss:",val_loss)

    def log_details(self, step, observation, action, next_observation):
        pass
