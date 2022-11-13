import tensorflow as tf
import numpy as np


celsius= np.array([-40,-10,0,8,15,22,38], dtype= float)
fahremheit = np.array ([-40,14,32,46,59,72,100], dtype= float)

layer= tf.keras.layers.Dense(units=1, input_shape=[1])
model= tf.keras.Sequential ([layer])

model.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss= "mean_squared_error" 
)

print ("start")
historial= model.fit(celsius,fahremheit, epochs=1000, verbose=False)
print ("done")

import matplotlib.pyplot as plt
plt.ylabel("loss magnitude")
plt.plot(historial.history["loss"])
#plt.show() #with 500 is ok
result= model.predict([100])
print (result)
