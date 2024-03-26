import numpy as np
import onnxruntime
from PIL import Image
import base64
import io
import kserve


class StyleModel(kserve.Model):
    def __init__(self, name: str, path: str):
        super().__init__(name)
        self.name = name
        self.path = path
        self.load()

    def load(self):
        self.model = onnxruntime.InferenceSession(self.path)
        self.ready = True

    def predict(self, payload, headers: dict = None) -> dict:
        data = payload["image"]

        raw_img_data = base64.b64decode(data)
        input_image = Image.open(io.BytesIO(raw_img_data)).convert('RGB')

        content_image = np.asarray(input_image, dtype=np.float32)
        content_image = content_image[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        model_input = {self.model.get_inputs()[0].name: content_image}
        model_output = self.model.run(None, model_input)

        result = model_output[0][0].clip(0, 255).transpose(1, 2, 0).astype('uint8')
        result_image = Image.fromarray(result, 'RGB')

        byte_arr = io.BytesIO()
        result_image.save(byte_arr, format='JPEG')
        encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')

        return {"predictions": encoded_img}


if __name__ == "__main__":
    model = StyleModel(name="style-model", path="/mnt/models/model_wave.onnx")
    model.load()
    kserve.ModelServer(workers=1).start([model])
