#!/usr/bin/env python

import asyncio

import json
import time
import websockets

def compute_res(x):
    nodes = {}
    adges = {}
    return x+1

async def client_connected_handler(websocket):
    print("Client connected")
    x = 0
    while True:
        
        message = await websocket.send(json.dumps({"nodes": {"r": x, "activity_state": 50} , "links": {"l": [1,x-1], "activity_state": 50}}))
        x = compute_res(x)
        time.sleep(1)
        # print(x)



async def main():
    async with websockets.serve(client_connected_handler, "", 5678):
        await asyncio.Future()  # run forever

def decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

if __name__ == "__main__":
    asyncio.run(main())



