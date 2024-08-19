import itertools
import numpy as np
import keras

'''
Data generator which yields batches of input windows and output values from whole profiles (useful for local learning of neural functionals).
'''

class DataGenerator(keras.utils.PyDataset):
    def __init__(self, simData, batch_size=32, steps_per_execution=1, shuffle=True, inputKeys=["rho"], paramsKeys=[], outputKeys=["c1"], windowSigma=2.0, filt=lambda sim: True, **kwargs):
        super().__init__(**kwargs)
        self.simData = {key: sim for key, sim in simData.items() if filt(sim)}
        print(f"Loaded {len(self.simData)} simulations")
        self.inputKeys = inputKeys
        self.paramsKeys = paramsKeys
        self.outputKeys = outputKeys
        self.windowSigma = windowSigma
        firstSimData = list(self.simData.values())[0]
        self.dz = 2 * firstSimData["profiles"]["z"][0]
        self.simDataBins = len(firstSimData["profiles"]["z"])
        self.windowBins = int(round(self.windowSigma/self.dz))
        self.validBins = {}
        self.inputDataPadded = {}
        for simId in self.simData.keys():
            valid = np.full(self.simDataBins, True)
            for k in self.outputKeys:
                valid = np.logical_and(valid, ~np.isnan(self.simData[simId]["profiles"][k]))
            self.validBins[simId] = np.flatnonzero(valid)
            self.inputDataPadded[simId] = np.pad(self.simData[simId]["profiles"][self.inputKeys], self.windowBins, mode="wrap")
        self.batch_size = batch_size
        self.steps_per_execution = steps_per_execution
        self.inputShape = (2*self.windowBins+1,)
        self.outputShape = (len(self.outputKeys),)
        self.shuffle = shuffle
        self.on_epoch_end()
        print(f"Initialized DataGenerator from {len(self.simData)} simulations which will yield up to {len(self.indices)} input/output samples in batches of {self.batch_size}")

    def __len__(self):
        return int(np.floor(len(self.indices) / (self.batch_size * self.steps_per_execution))) * self.steps_per_execution

    def __getitem__(self, index):
        ids = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        profiles = {key: np.empty((self.batch_size, *self.inputShape)) for key in self.inputKeys}
        params = {key: np.empty((self.batch_size, 1)) for key in self.paramsKeys}
        output = {key: np.empty((self.batch_size, *self.outputShape)) for key in self.outputKeys}
        for b, (simId, i) in enumerate(ids):
            for key in self.inputKeys:
                profiles[key][b] = self.inputDataPadded[simId][key][i:i+2*self.windowBins+1]
            for key in self.paramsKeys:
                params[key][b] = self.simData[simId]["params"][key]
            for key in self.outputKeys:
                output[key][b] = self.simData[simId]["profiles"][key][i]
        return (profiles | params), output

    def on_epoch_end(self):
        self.indices = []
        for simId in self.simData.keys():
            self.indices.extend(list(itertools.product([simId], list(self.validBins[simId]))))
        if self.shuffle == True:
            np.random.default_rng().shuffle(self.indices)

    def pregenerate(self):
        print("Pregenerating data from DataGenerator")
        batch_size_backup = self.batch_size
        self.batch_size *= len(self)
        data = self[0]
        self.batch_size = batch_size_backup
        return data

