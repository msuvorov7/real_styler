import argparse
import os
import sys
import zipfile


fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))


def create_archive(models_path: str) -> None:
    """
    Создать архив для отправки в S3
    :param models_path: путь до моделей
    :return:
    """
    with zipfile.ZipFile('serverless_functions.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(models_path + 'model_mosaic.onnx', 'models/model_mosaic.onnx')
        zf.write(models_path + 'model_udnie.onnx', 'models/model_udnie.onnx')
        zf.write(models_path + 'model_lines.onnx', 'models/model_lines.onnx')
        zf.write(models_path + 'model_candy.onnx', 'models/model_candy.onnx')
        zf.write(models_path + 'model_scream.onnx', 'models/model_scream.onnx')
        zf.write(models_path + 'model_night.onnx', 'models/model_night.onnx')
        zf.write(models_path + 'model_wave.onnx', 'models/model_wave.onnx')
        zf.write(fileDir + 'src/app/run.py', 'run.py')
        zf.write(fileDir + 'src/app/requirements.txt', 'requirements.txt')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--models_path', default='models/', dest='models_path')
    args = args_parser.parse_args()

    model_path = fileDir + args.models_path
    create_archive(model_path)
