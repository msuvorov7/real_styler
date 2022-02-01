import telebot
import torch
import os
from torch.autograd import Variable

from config.config import TOKEN, MODEL
from model.msg_net import Net
from model.utils import tensor_save_bgrimage, preprocess_batch, tensor_load_rgbimage, load_image, unloader
from model.nst_net import train_style_transfer, device, cnn, cnn_normalization_mean, cnn_normalization_std


print(TOKEN)
STAGE = ''


def set_stage(stage):
    global STAGE
    STAGE = stage


def get_stage():
    global STAGE
    return STAGE


bot = telebot.TeleBot(TOKEN)


@bot.message_handler(commands=['start'])
def welcome_start(message):
    bot.send_message(message.chat.id, 'Hello')


@bot.message_handler(commands=['help'])
def welcome_start(message):
    help_info = """
    /content -> download content image
    /style -> download style image
    /result -> get output image
    """
    bot.send_message(message.chat.id, help_info)


def is_files_exist():
    return os.path.exists('tmp/content.jpg') & os.path.exists('tmp/style.jpg')


@bot.message_handler(commands=["result"])
def handle_send_photo(message):
    if not is_files_exist():
        bot.send_message(message.chat.id, 'Download content and style images first')
        return
    if MODEL == 'MSG':
        content_image, style_image = prepare_inputs()
        get_output_image(content_image, style_image)
    elif MODEL == 'NST':
        content_img = load_image('tmp/content.jpg').to(device)
        style_img = load_image('tmp/style.jpg').to(device)
        input_img = content_img.clone()
        output = train_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                      content_img, style_img, input_img)
        unloader(output.squeeze(0)).save('tmp/output.jpg')
    print('Done')
    file = open('tmp/output.jpg', 'rb')
    bot.send_photo(message.chat.id, file)
    file.close()


@bot.message_handler(commands=["style"])
def change_stage_style(message):
    set_stage('s')
    bot.send_message(message.chat.id, 'Okay, download the style image')


@bot.message_handler(commands=["content"])
def change_stage_style(message):
    set_stage('c')
    bot.send_message(message.chat.id, 'Okay, download the content image')


def process_photo_message(message):
    if get_stage() == 's':
        filename = 'style'
    elif get_stage() == 'c':
        filename = 'content'
    else:
        print(STAGE)
        bot.send_message(message.chat.id, 'Download content and style images first')
        return
    print('message.photo =', message.photo)
    file_id = message.photo[-1].file_id
    print(filename)
    print('fileID =', file_id)
    file = bot.get_file(file_id)
    print('file.file_path =', file.file_path)
    downloaded_file = bot.download_file(file.file_path)

    with open(f"tmp/{filename}.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)


@bot.message_handler(content_types=['photo'])
def photo(message):
    process_photo_message(message)


def prepare_model():
    style_model = Net(ngf=128)
    model_dict = torch.load('21styles.model')
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)
    return style_model


def prepare_inputs():
    content_image = tensor_load_rgbimage('tmp/content.jpg', size=512,
                                         keep_asp=True).unsqueeze(0)
    style = tensor_load_rgbimage('tmp/style.jpg', size=512).unsqueeze(0)
    style = preprocess_batch(style)
    return content_image, style


def get_output_image(content_image, style):
    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)
    output = style_model(content_image)
    tensor_save_bgrimage(output.data[0], 'tmp/output.jpg', False)


style_model = prepare_model()

bot.polling()
