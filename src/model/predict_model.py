import numpy as np
import onnxruntime
from PIL import Image


if __name__ == '__main__':
    model = onnxruntime.InferenceSession('../../models/model_wave.onnx')
    content_image = np.asarray(Image.open('../../data/all.jpg').convert('RGB'), dtype=np.float32)
    content_image = content_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    model_input = {model.get_inputs()[0].name: content_image}
    model_output = model.run(None, model_input)

    result = model_output[0][0].clip(0, 255).transpose(1, 2, 0).astype('uint8')
    Image.fromarray(result).save('../../data/all_wave.jpg')
