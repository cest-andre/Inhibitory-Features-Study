@echo off

FOR /L %%X IN (0,8,255) DO (
python main_scatterplot.py --network alexnet --neuron %%X --layer .features.Conv2d10 --selected_layer 11 --main_layer_folder alexnet_layer5
python rf_size_activations.py --network alexnet --neuron %%X --layer .features.Conv2d10 --selected_layer 11 --main_layer_folder alexnet_layer5
python rf_size_activations.py --network alexnet --neuron %%X --layer .features.Conv2d10 --selected_layer 11 --main_layer_folder alexnet_layer5 --invert
)

FOR /L %%X IN (0,16,511) DO (
python main_scatterplot.py --network vgg16 --neuron %%X --layer .features.Conv2d17 --selected_layer 18 --main_layer_folder vgg_layer8
python rf_size_activations.py --network vgg16 --neuron %%X --layer .features.Conv2d17 --selected_layer 18 --main_layer_folder vgg_layer8
python rf_size_activations.py --network vgg16 --neuron %%X --layer .features.Conv2d17 --selected_layer 18 --main_layer_folder vgg_layer8 --invert
)

FOR /L %%X IN (0,16,511) DO (
python main_scatterplot.py --network vgg16 --neuron %%X --layer .features.Conv2d28 --selected_layer 29 --main_layer_folder vgg_layer13
python rf_size_activations.py --network vgg16 --neuron %%X --layer .features.Conv2d28 --selected_layer 29 --main_layer_folder vgg_layer13
python rf_size_activations.py --network vgg16 --neuron %%X --layer .features.Conv2d28 --selected_layer 29 --main_layer_folder vgg_layer13 --invert
)