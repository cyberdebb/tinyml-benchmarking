# count_macs.py  --  python count_macs.py models/cnn_classifier_resnet.keras
import sys

import keras
import numpy as np

model = keras.models.load_model(sys.argv[1])
macs = 0
for layer in model.layers:
    if isinstance(layer, (keras.layers.Conv1D, keras.layers.DepthwiseConv1D)):
        # each output sample = one MAC per kernel weight
        macs += layer.output.shape[1] * int(np.prod(layer.kernel.shape))
    elif isinstance(layer, keras.layers.Dense):
        macs += int(np.prod(layer.kernel.shape))

trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
print(f'Parameters: {model.count_params():,} (trainable: {trainable:,})')
print(f'MACs per inference: {macs:,}')