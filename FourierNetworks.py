import tensorflow as tf
import numpy as np
import math

# activation function CReLU
class CReLU(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def get_config(self):
        return {}

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        res = inputs[:,0]
        ims = inputs[:,1]

        return tf.stack([tf.nn.relu(res), tf.nn.relu(ims)], axis=1)

# Definition of Layers in frequnecy representation

# inverse fast Fourier transform
class IFFT(tf.keras.layers.Layer): 
    def __init__(self):
        super().__init__()

    def get_config(self):
        return {}

    def build(self, input_shape):
        pass

    def call(self, inputs): # (batch, cord, rows, cols, cha)
        res = inputs[:,0,:,:,:]
        ims = inputs[:,1,:,:,:]

        coms = tf.complex(res,ims)
        ffts = tf.signal.ifft3d(coms)
        reals = tf.math.real(ffts)

        return reals


# convolutional layer of SB-CNN, including Spectral Pooling (focused on the central region of centered Fourier transform)
class Dot(tf.keras.layers.Layer):
    def __init__(self, filters, act='crelu', out_factor=0.5):
        super().__init__()
        self.filters = filters
        self.act = CReLU()
        self.of = out_factor # factor de reduccion para Spectral Pooling

        if act == 'identity':
            self.act = tf.identity
    
    def get_config(self):
        return {"filters": self.filters, "act": self.act, "out_factor": self.of}

    def build(self, input_shape): # (batch, cord, row, cols, chan)
        # kernels
        self.W_r = self.add_weight(
            shape=(int(math.ceil(input_shape[-3]/2)), int(math.ceil(input_shape[-2]/2)), input_shape[-1], self.filters),
            initializer="random_normal", trainable=True)
        self.W_i = self.add_weight(
            shape=(int(math.ceil(input_shape[-3]/2)), int(math.ceil(input_shape[-2]/2)), input_shape[-1], self.filters),
            initializer="random_normal", trainable=True)
        
        # bias
        self.b_r = self.add_weight(shape=(self.filters,), initializer="zeros", trainable=True)
        self.b_i = self.add_weight(shape=(self.filters,), initializer="zeros", trainable=True)

        # dimension del spectral pooling
        self.r_min = int(math.ceil(input_shape[-3]*(1-self.of)/2)) # indice minimo
        self.r_max = int(math.ceil(input_shape[-3]*(1-self.of)/2)) + int(math.ceil(input_shape[-3]*(1-self.of))) # indice maximo
        self.c_min = int(math.ceil(input_shape[-2]*(1-self.of)/2)) # indice minimo
        self.c_max = int(math.ceil(input_shape[-2]*(1-self.of)/2)) + int(math.ceil(input_shape[-2]*(1-self.of))) # indice maximo

    def call(self, inputs): # x es de shape (batch,coordenadas,r,c,chanels)
        # spectral pooling
        sp = inputs[:,:,self.r_min:self.r_max,self.c_min:self.c_max,:]
        
        # convolucion
        x = tf.expand_dims(sp, axis=-1) # agregamos una dimension al final para hacer broadcasting

        # calculo de la parte real
        r = x[:,0]*self.W_r - x[:,1]*self.W_i  # aritmetica con broadcasting
        r = tf.reduce_sum(r, axis=3) + self.b_r# suma sobre los canales (convolucion suma sobre canales)

        # calculo de la parte imaginaria
        i = x[:,0]*self.W_i + x[:,1]*self.W_r  # aritmetica con broadcasting
        i = tf.reduce_sum(i, axis=3) + self.b_i # suma sobre los canales (convolucion suma sobre canales)

        # stack de las coordenadas
        y = tf.stack([r, i], axis=1)

        return self.act(y) # se evalua con la funcion de activacion


# auxiliar class for A and b variables definition
class RandomLowHigh(tf.keras.initializers.Initializer):
    def __init__(self):
        super().__init__()

    def __call__(self, shape, dtype=None, **kwargs): # shape (2, cha, filters)
        bias = tf.random.uniform(shape=(shape[1], shape[2]), minval=0, maxval=2, dtype=tf.int32) # [0] para magnitud y [1] para bias (esa coordenada ya esta bien)
        mag = (-1)**(bias)
        out = tf.stack([mag, bias], axis=0)
        return tf.cast(out, tf.float32)


# convolutional layer implementing the Butterworth filters
class ButterworthLayer(tf.keras.layers.Layer):
    def __init__(self, filters, norm=1.0, es=0.45, n=2, act='crelu'):
        super().__init__()
        self.filters = filters
        self.norm = norm # los valores en el grid quedan entre -norm y norm

        # frecuencia
        self.es = es
        self.n = n

        self.act = CReLU()

        if act == 'identity':
            self.act = tf.identity
    
    def get_config(self):
        return {'filters': self.filters, 'norm': self.norm, 'es': self.es, 'n': self.n, 'act': self.act}

    def build(self, input_shape): # (batch, cord, rows, cols, cha)    
        # definicion del grid
        dx0 = (self.norm-(-self.norm))/(input_shape[-2]-1)
        dy0 = (self.norm-(-self.norm))/(input_shape[-3]-1)
        self.x = tf.range(-1*self.norm, self.norm+0.0001, delta=dx0, dtype=tf.float32)
        self.y = tf.range(-1*self.norm, self.norm+0.0001, delta=dy0, dtype=tf.float32)
        self.x, self.y = tf.meshgrid(self.x, self.y)

        # se expande la dimension dos veces para trabajar con canales y filtros (rows, cols, 1, 1)
        self.x = tf.expand_dims(self.x, axis=-1)
        self.y = tf.expand_dims(self.y, axis=-1)
        self.x = tf.expand_dims(self.x, axis=-1)
        self.y = tf.expand_dims(self.y, axis=-1)

        self.A_b = self.add_weight(shape=(2,input_shape[-1], self.filters), initializer=RandomLowHigh(), name='A', trainable=False)
        self.x0 = self.add_weight(shape=(input_shape[-1], self.filters), name='x0', 
                                  initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.25), trainable=True)
        self.y0 = self.add_weight(shape=(input_shape[-1], self.filters), name='y0', 
                                  initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.25), trainable=True)

    def call(self, inputs): # k sera (rows, cols, cha, filters), inputs (bathc, cord, row, cols, cha)
        # se construyen los filtros Gaussianos (se define una vez por batch, entonces a la larga no es tan costoso)
        ks = self.A_b[0]/(1+(((self.x-self.x0)**2+(self.y-self.y0)**2)/(self.es**2))**self.n)+self.A_b[1]

        # producto de Hadamard
        inputs = tf.expand_dims(inputs, axis=-1) # dimension adicional para los nuevos canales
        y = inputs*ks
        y = tf.reduce_mean(y, axis=4)

        return self.act(y)


# implementation of Spectral Average Pooling
class Spect_Avg_Pool(tf.keras.layers.Layer):
    def __init__(self, size=2): # por defecto reduce la dimension ala mitad
        super().__init__()
        self.size = size

    def get_config(self):
        return {'size': self.size}

    def build(self, input_shape): # considera una dimension (batch, 2, n*n)
        self.mp_res = tf.keras.layers.AvgPool2D(pool_size=(self.size,self.size), strides=(self.size,self.size))
        self.mp_ims = tf.keras.layers.AvgPool2D(pool_size=(self.size,self.size), strides=(self.size,self.size))

    def call(self, x):
        res = x[:,0]
        ims = x[:,1]

        res = self.mp_res(res)
        ims = self.mp_ims(ims)

        # stack de las coordenadas
        y = tf.stack([res, ims], axis=1)

        return y