import random
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from keras.optimizers import Adam

""" Helper to set seeds for consistency """
def set_seeds(seed = 100):
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)

""" Balance Classes (upward or downward)"""
def class_weights(df):
    if "dir" not in df.columns:
        raise ValueError("DataFrame must contain a 'dir' column for class_weights.")
    
    dir_values = df["dir"].astype(int) # Ensure int for bincount
    
    if dir_values.empty: # Handle empty DataFrame/Series
        return {0: 1.0, 1: 1.0} 

    counts = np.bincount(dir_values)
    
    class0_count = counts[0] if len(counts) > 0 else 0
    class1_count = counts[1] if len(counts) > 1 else 0
        
    total_samples = len(dir_values) # Total samples in the 'dir' column

    if class0_count == 0 and class1_count == 0: # No relevant classes found
        return {0: 1.0, 1: 1.0}

    # Calculate weights for classes that are present
    # Formula: (total_samples / (num_classes * count_for_that_class))
    # Here, num_classes = 2 (binary classification)
    calculated_weights = {}
    if class0_count > 0:
        calculated_weights[0] = total_samples / (2.0 * class0_count)
    if class1_count > 0:
        calculated_weights[1] = total_samples / (2.0 * class1_count)
    
    # If, for some reason, calculated_weights is empty (e.g., dir_values contained only NaNs that became 0s),
    # provide default weights. This check might be redundant given earlier ones.
    if not calculated_weights:
        return {0: 1.0, 1: 1.0}
        
    return calculated_weights
  

optimizer = Adam(learning_rate = 0.0001)

""" Create the Model """
def create_model(hidden_layers = 2, layer_units = 100, dropout = False, rate = 0.3, regularize = False, reg = l1(0.0005), optimizer = optimizer, input_dim = None):
  if not regularize:
    reg = None
  model = Sequential()
  model.add(Dense(layer_units, input_dim = input_dim, activity_regularizer=reg, activation="relu"))
  if dropout:
    model.add(Dropout(rate, seed = 100))
  for layer in range(hidden_layers):
    model.add(Dense(layer_units, activation="relu", activity_regularizer=reg))
    if dropout:
      model.add(Dropout(rate, seed=100))
  model.add(Dense(1,activation="sigmoid"))
  model.compile(loss = "binary_crossentropy", optimizer= optimizer, metrics=["accuracy"])
  return model