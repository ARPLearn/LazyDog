#!/usr/bin/env/python
# File name   : server.py
# Production  : Upper Ctrl for Robots
# Author      : WaveShare
from threading import Thread
import time
import threading
import os
import socket
import info
import asyncio
import websockets
import json
import app
import pygame
import pyttsx3
from queue import Queue
import threading

ipaddr_check = "192.168.4.1"
flask_app = None  # Global variable
engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Speed
engine.setProperty('volume', 0.9)  # Volume
speech_queue = Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:  # Signal to stop thread
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {e}")
        speech_queue.task_done()

# Start speech thread right after engine initialization
speech_thread = threading.Thread(target=speech_worker)
speech_thread.daemon = True
speech_thread.start()

def ap_thread():
    os.system("sudo create_ap wlan0 eth0 WAVE_BOT 12345678")

def wifi_check():
    global ipaddr_check
    time.sleep(5)
    try:
        s = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(("1.1.1.1",80))
        ipaddr_check = s.getsockname()[0]
        s.close()
        print(ipaddr_check)
    except:
        ap_threading = threading.Thread(target=ap_thread)   
        ap_threading.setDaemon(True)                     
        ap_threading.start()

async def check_permit(websocket):
    while True:
        try:
            recv_str = await websocket.recv()
            cred_dict = recv_str.split(":")
            if cred_dict[0] == "admin" and cred_dict[1] == "123456":
                response_str = "Connected!"
                await websocket.send(response_str)
                return True
            else:
                response_str = "sorry, the username or password is wrong, please submit again"
                await websocket.send(response_str)
        except Exception as e:
            print(f"Auth error: {e}")
            return False

async def recv_msg(websocket):
    global flask_app
    while True:
        try:
            response = {
                'status': 'ok',
                'title': '',
                'data': None
            }

            data = await websocket.recv()
            print(f"Received: {data}")
            
            try:
                data = json.loads(data)
            except:
                pass

            if data:
                if isinstance(data, str):
                    flask_app.commandInput(data)
                    
                    if 'get_info' == data:
                        response['title'] = 'get_info'
                        response['data'] = [info.get_cpu_tempfunc(), info.get_cpu_use(), info.get_ram_info()]
                        
                    elif data.startswith('speak:'):
                        text = data.replace('speak:', '').strip()
                        try:
                            speech_queue.put(text)  # Add to queue instead of speaking directly
                            response['title'] = 'speak'
                            response['data'] = 'Speech queued'
                        except Exception as e:
                            response['status'] = 'error'
                            response['data'] = str(e)
                        
                    elif 'bark' == data:
                        try:
                            pygame.mixer.init()
                            sound_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sounds', 'bark.mp3')
                            
                            if os.path.exists(sound_file):
                                pygame.mixer.music.load(sound_file)
                                pygame.mixer.music.play()
                                time.sleep(0.1)
                                response['title'] = 'bark'
                                response['data'] = 'Bark initiated'
                            else:
                                response['status'] = 'error'
                                response['data'] = 'Sound file not found'
                        except Exception as e:
                            response['status'] = 'error'
                            response['data'] = str(e)

                    elif 'CVFLColorSet' in data:
                        color = int(data.split()[1])
                        flask_app.camera.colorSet(color)

                    elif 'CVFLL1' in data:
                        pos = int(data.split()[1])
                        flask_app.camera.linePosSet_1(pos)

                    elif 'CVFLL2' in data:
                        pos = int(data.split()[1])
                        flask_app.camera.linePosSet_2(pos)

                    elif 'CVFLSP' in data:
                        err = int(data.split()[1])
                        flask_app.camera.errorSet(err)

                elif isinstance(data, dict):
                    if data['title'] == "findColorSet":
                        color = data['data']
                        flask_app.colorFindSet(color[0], color[1], color[2])

            print(f"Sending response: {response}")
            await websocket.send(json.dumps(response))
        except Exception as e:
            print(f"Error in recv_msg: {e}")
            break

def run_flask():
    global flask_app
    flask_app = app.webapp()
    flask_app.startthread()
    flask_app.sendIP(ipaddr_check)

async def main_logic(ws, path=None):
    print(f"New connection attempt")
    try:
        await check_permit(ws)
        print("Client authenticated")
        await recv_msg(ws)
    except Exception as e:
        print(f"Connection error: {e}")
        try:
            await ws.close()
        except:
            pass

async def run_websocket():
    async with websockets.serve(main_logic, '0.0.0.0', 8888):
        print('Websocket server started on port 8888')
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    print("1. Starting wifi check...")
    wifi_check()
    
    print("2. Starting Flask in thread...")
    flask_thread = Thread(target=run_flask)
    flask_thread.start()
    
    print("3. Starting websocket server...")
    asyncio.run(run_websocket())
