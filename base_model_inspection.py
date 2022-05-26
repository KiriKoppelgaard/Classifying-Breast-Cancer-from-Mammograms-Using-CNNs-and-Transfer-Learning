'''
Script to inspect the base models, inceptionv3 and efficientnetv2S.
It is used to find layer numbers of the layer we need to freeze from, so we can freeze the two top blocks of the base models.

Was run interactively. 
'''

from tensorflow.keras.applications import InceptionV3, EfficientNetV2S
from contextlib import redirect_stdout
from keras.utils.vis_utils import plot_model

# shape of input images
input_shape = (100,100,3)

# define base models for inspection
inceptionv3_base = InceptionV3( 
    input_shape=input_shape, 
    weights="imagenet", # include pre-trained weights
    include_top=False) # don't include top/last fully connected layer

efficientnetv2_base = EfficientNetV2S(
                input_shape=input_shape,
                weights='imagenet', # include pre-trained weights from training on imagenet
                include_top=False) # don't include top/last fully connected layer

# plot model architecture
plot_model(inceptionv3_base, f'output/inceptionv3/inceptionv3_base_architecture.png', show_shapes=True)
plot_model(efficientnetv2_base, f'output/efficientnetv2s/efficientnetv2s_base_architecture.png', show_shapes=True)

# visualize layer names and layer indices 
# used together with the plot_model png to figure out how many layers we should freeze in order to unfreeze the top two blocks 
for i, layer in enumerate(inceptionv3_base.layers):
    print(i, layer.name)

for i, layer in enumerate(efficientnetv2_base.layers): #[:434]):
    print(i, layer.name)

#print(efficientnetv2_base.layers[433])

# save summaries
with open(f'output/inceptionv3/inceptionv3_base_summary.txt', 'w') as f:
    with redirect_stdout(f):
        inceptionv3_base.summary()

with open(f'output/efficientnetv2s/efficientnetv2S_base_summary.txt', 'w') as f:
    with redirect_stdout(f):
        efficientnetv2_base.summary()

# print number of layers
print("number of layers of inceptionv3_base:", len(inceptionv3_base.layers))
print("number of layers of inceptionv3_base:", len(efficientnetv2_base.layers))

