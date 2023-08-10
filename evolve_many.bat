@echo off

FOR /L %%X IN (0,16,63) DO (
python evolve_test.py --network resnet18 --neuron %%X --layer_name .layer1.BasicBlock0 --neuron_coord 27
@REM python evolve_test.py --network resnet18 --neuron %%X --layer_name .layer4.BasicBlock1 --neuron_coord 3
)

@REM python evolve_test.py --network resnet18 --neuron 0 --layer_name .layer2.1.Conv2dconv2 --neuron_coord 13
@REM python evolve_test.py --network resnet18 --neuron 109 --layer_name .layer2.1.Conv2dconv2 --neuron_coord 13
@REM python evolve_test.py --network resnet18 --neuron 65 --layer_name .layer2.1.Conv2dconv2 --neuron_coord 13
