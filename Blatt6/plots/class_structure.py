import numpy as np
import pandas as pd
### one classy boi 
class KNN:
	'''KNN Classifier.
	
	Attributes
	----------
	k : int
	    Number of neighbors to consider.
	'''
	def __init__(self, k):
	    '''Initialization.
	    Parameters are stored as member variables/attributes.
	    
	    Parameters
	    ----------
	    k : int
	        Number of neighbors to consider.
	    '''
	    self.k = k
	    #Hier fehlt glaube ich noch ein self.traindata und self.trainlabels
	def fit(self, X, y):
	    '''Fit routine.
	    Training data is stored within object.
	    
	    Parameters
	    ----------
	    X : numpy.array, shape=(n_samples, n_attributes)
	        Training data.
	    y : numpy.array shape=(n_samples)
	        Training labels.
	    '''
	    # Code
	    self.traindata = X
	    self.trainlabels = y	
	
	def predict(self, X):
		'''Prediction routine.
		Predict class association of each sample of X.
		
		Parameters
		----------
		X : numpy.array, shape=(n_samples, n_attributes)
		    Data to classify.
		
		Returns
		-------
		prediction : numpy.array, shape=(n_samples)
		    Predictions, containing the predicted label of each sample.
		'''
		prediction = np.array([])
		# Code
		for x in X:
			abst1 = self.traindata.T[0] - x.T[0] 
			abst2 = self.traindata.T[1] - x.T[1] 
			abst3 = self.traindata.T[2] - x.T[2]
			abst = np.sqrt(abst1**2 + abst2**2 + abst3**2) 
			indis = np.argsort(abst)
			indis = indis[:self.k]
			labels = np.array(self.trainlabels[indis])  
			u, inds = np.unique(labels, return_inverse=True)
			label = u[np.argmax(np.bincount(inds))]
			prediction = np.append(prediction,label)
		return prediction	
###read in some data
Background = pd.read_hdf('NeutrinoMC.hdf5', key= 'Background')
Signal = pd.read_hdf('NeutrinoMC.hdf5', key= 'Signal')
### Some training data
TrainingsetX = np.ndarray(shape=(5000,3))
TrainingsetX.T[0] = np.append(Signal.x[:1666],Background.x[:3334])
TrainingsetX.T[1] = np.append(Signal.y[:1666],Background.y[:3334])
TrainingsetX.T[2] = np.append(Signal.NumberOfHits[:1666],Background.NumberOfHits[:3334])
TrainingsetY = np.append(np.ones(1666),np.zeros(3334))
### data to do some KNN on 
n1dataX = np.ndarray(shape=(30000,3))
n1dataX.T[0] = np.append(Signal.x[:10000],Background.x[:20000])
n1dataX.T[1] = np.append(Signal.y[:10000],Background.y[:20000])
n1dataX.T[2] = np.append(Signal.NumberOfHits[:10000],Background.NumberOfHits[:20000])

n2dataX = np.ndarray(shape=(30000,3))
n2dataX.T[0] = n1dataX.T[0] 
n2dataX.T[1] = n1dataX.T[1] 
n2dataX.T[2] = np.append(np.log10(Signal.NumberOfHits[:10000]),np.log10(Background.NumberOfHits[:20000]))

##### KNN on Data 
Neutrino1 = KNN(10)
Neutrino1.fit(TrainingsetX,TrainingsetY)
N1labels = Neutrino1.predict(n1dataX)

Neutrino2 = KNN(10)
Neutrino2.fit(TrainingsetX,TrainingsetY)
N2labels = Neutrino2.predict(n2dataX)

Neutrino3 = KNN(20)
Neutrino3.fit(TrainingsetX,TrainingsetY)
N3labels = Neutrino3.predict(n1dataX)

#### reinheit Effizienz und Signifikanz
def reineffisign(predictions):
	S = predictions[predictions ==1 ]
	B = predictions[predictions ==0 ]
	array1 = predictions[:10000] #sollte nur Signal sein
	array2 = predictions[10000:] #sollte nur Untergrund sein
	trupos = len(array1[array1 == 1]) #alle Signale zu S	
	truneg = len(array2[array2 == 0]) # Back das zu B 
	falpos = len(array2[array2 == 1]) #Back das zu S	
	falneg = len(array1[array1 ==0])  # Signal zu B
	rein = (trupos/(truneg + falpos))
	effi = (trupos/(trupos + falneg))
	sign = (len(S) / np.sqrt(len(S) + len(B))) 
	return rein,effi,sign

print(reineffisign(N1labels))
print(reineffisign(N2labels))
print(reineffisign(N3labels))


