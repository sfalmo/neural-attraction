import numpy as np
import tensorflow as tf
import keras

from utils import DataGenerator

tf.config.experimental.enable_tensor_float_32_execution(False)


'''
Load and filter data
'''

def filt(sim):
    T = sim["params"]["T"]
    return T > 1.0

simData = np.load("data/LJ.npy", allow_pickle=True).item()

trainingGenerator = DataGenerator(simData, batch_size=256, windowSigma=3.5, inputKeys=["rho"], paramsKeys=["T", "L_inv"], outputKeys=["c1"], filt=filt)


'''
Define model
'''

profileInputs = {"rho": keras.Input(shape=trainingGenerator.inputShape, name="rho")}
paramsInputs = {paramKey: keras.Input(shape=(1,), name=paramKey) for paramKey in ["T", "L_inv"]}

x = keras.layers.Concatenate()(list((profileInputs | paramsInputs).values()))
x = keras.layers.Dense(512, activation="softplus")(x)
x = keras.layers.Dense(512, activation="softplus")(x)
x = keras.layers.Dense(512, activation="softplus")(x)
x = keras.layers.Dense(512, activation="softplus")(x)
outputs = {"c1": keras.layers.Dense(trainingGenerator.outputShape[0], name="c1")(x)}

model = keras.Model(inputs=(profileInputs | paramsInputs), outputs=outputs)
optimizer = keras.optimizers.Adam()
loss = keras.losses.MeanSquaredError()
metrics = [keras.metrics.MeanAbsoluteError()]
model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics,
)
model.summary()


'''
Prepare data pipeline
'''

def gen():
    for i in range(len(trainingGenerator)):
        yield trainingGenerator[i]

train_dataset = tf.data.Dataset.from_generator(gen, output_signature=(
    {
        "rho": tf.TensorSpec(shape=(trainingGenerator.batch_size, trainingGenerator.inputShape[0]), dtype=tf.float32),
        "T": tf.TensorSpec(shape=(trainingGenerator.batch_size, 1), dtype=tf.float32),
        "L_inv": tf.TensorSpec(shape=(trainingGenerator.batch_size, 1), dtype=tf.float32),
    },
    {
        "c1": tf.TensorSpec(shape=(trainingGenerator.batch_size, 1), dtype=tf.float32),
    }
)).prefetch(tf.data.AUTOTUNE)


'''
Do training
'''

model.fit(
    train_dataset,
    epochs=200,
    callbacks=[
        keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * 0.92),
        keras.callbacks.ModelCheckpoint(filepath="models/current.keras"),
    ]
)
