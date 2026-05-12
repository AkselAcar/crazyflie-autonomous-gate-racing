#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2021 Bitcraze AB
#
#  AI-deck demo
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License along with
#  this program; if not, write to the Free Software Foundation, Inc., 51
#  Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#  Demo for showing streamed JPEG images from the AI-deck example.
#
#  By default this demo connects to the IP of the AI-deck example when in
#  Access point mode.
#
#  The demo works by opening a socket to the AI-deck, downloads a stream of
#  JPEG images and looks for start/end-of-frame for the streamed JPEG images.
#  Once an image has been fully downloaded it's rendered in the UI.
#
#  Note that the demo firmware is continously streaming JPEG files so a single
#  JPEG image is taken from the stream using the JPEG start-of-frame (0xFF 0xD8)
#  and the end-of-frame (0xFF 0xD9).

import argparse
import time
import socket
import struct
import numpy as np
import cv2
import os
from datetime import datetime

# Args for setting IP/port of AI-deck. Default settings are for when
# AI-deck is in AP mode.
parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
parser.add_argument("-n", default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
parser.add_argument('--save', action='store_true', help="Save streamed images")
args = parser.parse_args()


if args.save:
    # Crée une chaîne de caractères avec la date et l'heure (ex: "2026-05-04_11-27-11")
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
    # On définit les chemins des dossiers
    path_raw = f"stream_out/{timestamp}/raw"
    path_color = f"stream_out/{timestamp}/color"
    
    # On crée les dossiers (exist_ok=True évite une erreur si le dossier existe déjà)
    os.makedirs(path_raw, exist_ok=True)
    os.makedirs(path_color, exist_ok=True)
    
    print(f"Les images seront sauvegardées dans : stream_out/{timestamp}/")

LOCAL_PORT  = 5001
START_MAGIC = b'FER'
CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
IMG_WIDTH        = 324
IMG_HEIGHT       = 244
MIN_JPEG_BYTES   = 5000

print("Opening UDP socket, waiting for AI-deck {}:{}...".format(args.n, args.p))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
client_socket.bind(('0.0.0.0', LOCAL_PORT))
client_socket.sendto(START_MAGIC, (args.n, args.p))
print("UDP socket ready, stream started")

start = time.time()
count = 0
imgStream = bytearray()
expected_size = 0
receiving = False
img_format = 0

try:
    while True:
        data, _ = client_socket.recvfrom(2048)
        if len(data) < CPX_HEADER_SIZE:
            continue
        payload = data[CPX_HEADER_SIZE:]

        # Image header → start of new frame
        if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
            [magic, width, height, depth, fmt, size] = struct.unpack('<BHHBBI', payload[:IMG_HEADER_SIZE])
            # 'magic' is a magic byte saying that the packet is a bitcraze image
            # 'depth' = nbr of bytes per pixel
            # 'format' = 0 if RAW Bayer, other if JPEG
            # 'size' = image size in bytes
            if width == IMG_WIDTH and height == IMG_HEIGHT and 0 < size < 65536:
                expected_size = size
                imgStream = bytearray()
                receiving = True
                img_format = fmt
            continue

        if not receiving:
            continue

        imgStream.extend(payload)

        if len(imgStream) >= expected_size:
            receiving = False
            count += 1
            meanTimePerImage = (time.time() - start) / count
            print(f"Mean: {meanTimePerImage:.4f}s | FPS: {1/meanTimePerImage:.2f}")

            """
            BAYER = Raw, best quality, no computation onboard, heavier to send
            JPEG  = already compressed onboard (so more computation for drone), lower quality, 5-20x lighter (easier to send)
            """
            if img_format == 0:
                bayer_img = np.frombuffer(imgStream, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH))
                color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
                cv2.imshow('Raw', bayer_img)
                cv2.imshow('Color', color_img)
                if args.save:
                    cv2.imwrite(f"{path_raw}/img_{count:06d}.png", bayer_img)
                    cv2.imwrite(f"{path_color}/img_{count:06d}.png", color_img)
                cv2.waitKey(1)
            else:
                soi = imgStream.find(b'\xff\xd8')
                eoi = imgStream.rfind(b'\xff\xd9')
                if soi >= 0 and eoi > soi and (eoi + 2 - soi) >= MIN_JPEG_BYTES:
                    with open("img.jpeg", "wb") as f:
                        f.write(imgStream[soi:eoi + 2])
                    nparr = np.frombuffer(imgStream, np.uint8, count=eoi + 2 - soi, offset=soi)
                    decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                    if args.save and decoded is not None:
                        cv2.imwrite(f"{path_raw}/img_{count:06d}.png", decoded)
                    #cv2.imshow('JPEG', decoded)
                cv2.waitKey(1)

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    client_socket.close()
    cv2.destroyAllWindows()