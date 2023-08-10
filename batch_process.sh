#!/bin/bash


for i in {10..41..1}
do
  python tuning_curve.py --network resnet18 --neuron $i --layer .layer4.1.Conv2dconv2 --selected_layer 63 --inh_abl
  python tuning_curve.py --network resnet18-untrained --neuron $i --layer .layer4.1.Conv2dconv2 --selected_layer 63 --inh_abl
  python tuning_curve.py --network resnet18-robust --neuron $i --layer .layer4.1.Conv2dconv2 --selected_layer 63 --inh_abl
done

#for i in {42..63..1}
#do
#  python evolve_test.py --network resnet18 --neuron $i --layer_name .layer4.1.Conv2dconv2 --neuron_coord 3 --type resnet18-untrained
#
#  python tuning_curve.py --network resnet18 --neuron $i --layer .layer4.1.Conv2dconv2 --selected_layer 63
#  python tuning_curve.py --network resnet18 --neuron $i --layer .layer4.1.Conv2dconv2 --selected_layer 63 --inh_abl
#
#  python tuning_curve.py --network resnet18-untrained --neuron $i --layer .layer4.1.Conv2dconv2 --selected_layer 63
#  python tuning_curve.py --network resnet18-untrained --neuron $i --layer .layer4.1.Conv2dconv2 --selected_layer 63 --inh_abl
#
#  python tuning_curve.py --network resnet18-robust --neuron $i --layer .layer4.1.Conv2dconv2 --selected_layer 63
#  python tuning_curve.py --network resnet18-robust --neuron $i --layer .layer4.1.Conv2dconv2 --selected_layer 63 --inh_abl
#done
