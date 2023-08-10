from thingsvision import get_extractor
from torchvision import transforms
import torch
from PIL import Image

model_name = 'alexnet'
source = 'torchvision'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
neuron_coord = None
conv_layer = True
if conv_layer:
    neuron_coord = 6
module_name = 'features.10'
unit_id = 25

extractor = get_extractor(
    model_name=model_name,
    source=source,
    device=device,
    pretrained=True
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
norm_trans = transforms.Normalize(mean=MEAN, std=STD)
imnet_trans = transforms.Compose([
    transforms.ToTensor(),
    norm_trans
])

exc_prototype = Image.open(r"/home/andre/evolved_data/alexnet_.features.Conv2d10_unit25/exc/Best_Img_alexnet_.features.Conv2d10_7.png")
exc_prototype = imnet_trans(exc_prototype)

inh_component = Image.open(r"/home/andre/evolved_data/alexnet_.features.Conv2d10_unit26/inh/Best_Img_alexnet_.features.Conv2d10_0.png")
inh_component = imnet_trans(inh_component)
# inh_component = torch.ones(exc_prototype.shape)
# inh_component = norm_trans(inh_component)

mixed_img = exc_prototype
# mixed_img = inh_component
# mixed_img = (exc_prototype + inh_component) / 2
mixed_img = torch.unsqueeze(mixed_img, 0)
mixed_img = torch.unsqueeze(mixed_img, 0)

features = extractor.extract_features(
    batches=mixed_img,
    module_name=module_name,
    flatten_acts=False
)

scores = None
if neuron_coord is not None:
    scores = features[:, unit_id, neuron_coord, neuron_coord]
else:
    scores = features[:, unit_id]

print(scores)