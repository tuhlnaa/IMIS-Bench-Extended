import sys
import numpy as np
from PIL import Image
from pathlib import Path



# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import parse_args
from src.utils.inference import determine_device, load_model
from segment_anything.predictor import ImagePreprocessor

def main():
    """Test the network initialization."""
    config = parse_args()

    image_path = r"./data/samples/ABD_001_67.png"
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    device = determine_device(config)
    imis_net, predictor = load_model(config, device)

    X = ImagePreprocessor(image_size=config.model.image_size, model_image_format="RGB")
    transform = X.get_transforms()
    
    input_tensor = transform(image_array).unsqueeze(0)
    
    output = imis_net(input_tensor)
    print(output.dtype)
    print(output.shape)


if __name__ == '__main__':
    main()