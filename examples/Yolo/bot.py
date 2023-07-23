#!/usr/bin/env python3
""" Implements a bot for sending messages with images """

import telegram
import io
import os
import configparser
from datetime import datetime


def read_serialize_imagefile (file_name):
    with open(file_name, 'rb') as stream :
        serialize_image(stream.read())

def serialize_image( image ):
    """
    Returns a serialized version of an image in bytes format.
    :image: A bytes object representing an image.
    :returns: A io serialized object representing the image.
    """

    return io.BufferedReader(io.BytesIO(image))

class bot:
    def __init__(self, send_interval=60):
        config_parser = configparser.ConfigParser()
        bot_home_path = os.path.dirname( os.path.realpath(__file__))
        config_parser.read( bot_home_path + '/conf.ini')

        bot_token = config_parser.get('MAIN','Bot_token')
        self.chat_id = int(config_parser.get('MAIN','Chat_id'))
        self.bot = telegram.Bot(bot_token)
        self.home = bot_home_path
        self.send_interval = send_interval
        self.previous_send_time = datetime.now() # TODO: make prevent start interval

    def send_image(self, text, image_file):
        #self.bot.sendMessage(chat_id=chat_id, text=text)
        now = datetime.now()
        diff_time = now - self.previous_send_time
        if diff_time.total_seconds() > self.send_interval:
            self.bot.sendPhoto(chat_id=self.chat_id, photo = open( image_file, 'rb'),caption=text)
            self.previous_send_time = datetime.now()
#        self.bot.sendPhoto(chat_id=self.chat_id, photo = open( self.home + '/sample_image.jpg', 'rb'),caption=text)
#`        self.bot.sendPhoto(chat_id=self.chat_id, photo = read_serialize_imagefile(image),caption=text)
#
#if __name__ == '__main__':
#    bot = bot()
#    bot.send_image("Hallo, dit is een bot", 'sample_image.jpg')

