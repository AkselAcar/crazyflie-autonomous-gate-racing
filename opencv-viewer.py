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

# Args for setting IP/port of AI-deck. Default settings are for when
# AI-deck is in AP mode.
parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
parser.add_argument("-n", default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default=5000, metavar="port", help="AI-deck port")
parser.add_argument('--save', action='store_true', help="Save streamed images")
args = parser.parse_args()

if args.save:
    os.makedirs("stream_out/raw", exist_ok=True)
    os.makedirs("stream_out/debayer", exist_ok=True)

print("Connecting to socket on {}:{}...".format(args.n, args.p))
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((args.n, args.p)) # opens a connection on which the drone will send data (pictures here)
print("Socket connected")

def rx_bytes(size):
    """
    Reads exactly 'size' bytes from the socket. The 'while' is needed because 'recv' does not
    guarantee to return the number of bytes asked in one time.
    """
    data = bytearray()
    while len(data) < size:
        chunk = client_socket.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed by remote")
        data.extend(chunk)
    return data

start = time.time()
count = 0

try:
    while True:
        # First get the info
        packetInfoRaw = rx_bytes(4) # this packet gives info on the packet of imgHeader, that gives info on the image
        #print(packetInfoRaw)
        # each message begins by 'length' (2 bytes) = size of the rest of the packet
        # then 'routing' (1 byte) gives source and destination
        # then 'function' (1 byte) gives the function type
        [length, routing, function] = struct.unpack('<HBB', packetInfoRaw)
        #print("Length is {}".format(length))
        #print("Route is 0x{:02X}->0x{:02X}".format(routing & 0xF, routing >> 4))
        #print("Function is 0x{:02X}".format(function))

        imgHeader = rx_bytes(length - 2) # minus 2 because 'dst' and 'src' are counted in 'length' but already read
        #print(imgHeader)
        #print("Length of data is {}".format(len(imgHeader)))
        [magic, width, height, depth, format, size] = struct.unpack('<BHHBBI', imgHeader)
        # 'magic' is a magic byte saying that the packet is a bitcraze image
        # 'depth' = nbr of bytes per pixel
        # 'format' = 0 if RAW Bayer, other if JPEG
        # 'size' = image size in bytes
        if magic == 0xBC:
            #print("Magic is good")
            #print("Resolution is {}x{} with depth of {} byte(s)".format(width, height, depth))
            #print("Image format is {}".format(format))
            #print("Image size is {} bytes".format(size))

            # Now we start rx the image, this will be split up in packages of some size
            imgStream = bytearray()
            while len(imgStream) < size: # we get back the image in several packets
                packetInfoRaw = rx_bytes(4) # this packet is a part of the image
                [length, dst, src] = struct.unpack('<HBB', packetInfoRaw)
                # 'length' (2 bytes) = size of this chunk
                # 'dst' (1 byte) = destination
                # 'src' (1 byte) = source
                #print("Chunk size is {} ({:02X}->{:02X})".format(length, src, dst))
                imgStream.extend(rx_bytes(length - 2)) # minus 2 because 'dst' and 'src' are counted in 'length' but already read

            count += 1
            meanTimePerImage = (time.time() - start) / count
            print(f"Mean: {meanTimePerImage:.4f}s | FPS: {1/meanTimePerImage:.2f}")
            
            """
            BAYER = Raw, best quality, no computation onboard, heavier to send
            JPEG  = already compressed onboard (so more computation for drone), lower quality, 5-20x lighter (easier to send)
            """
            if format == 0:
                bayer_img = np.frombuffer(imgStream, dtype=np.uint8).reshape((244, 324)) # np.frombuffer is flattened, we reshape in 2D
                color_img = cv2.cvtColor(bayer_img, cv2.COLOR_BayerBG2BGRA)
                cv2.imshow('Raw', bayer_img)
                cv2.imshow('Color', color_img)
                if args.save:
                    cv2.imwrite(f"stream_out/raw/img_{count:06d}.png", bayer_img)
                    cv2.imwrite(f"stream_out/debayer/img_{count:06d}.png", color_img)
                cv2.waitKey(1)
            else:
                with open("img.jpeg", "wb") as f:
                    f.write(imgStream)
                nparr = np.frombuffer(imgStream, np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                cv2.imshow('JPEG', decoded)
                cv2.waitKey(1)

except KeyboardInterrupt:
    print("Stopped by user")
except ConnectionError as e:
    print(f"Connection error: {e}")
finally:
    client_socket.close()
    cv2.destroyAllWindows()