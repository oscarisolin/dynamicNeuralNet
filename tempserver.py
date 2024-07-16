#!/usr/bin/env python

import asyncio

import json
import time
import websockets
import random


add_or_remove = [0, 1]

random.sample(add_or_remove, k=1)
amount = 10


def get_activ():
    return random.random()*100

async def client_connected_handler(websocket):
    print("Client connected")
    while True:
        data = [[],[]]

        nodenumber = random.sample(range(1, amount), k=1)[0]
        linknumber = random.sample(range(1, amount), k=1)[0]
        noderange = range(0, nodenumber)
        linkrange = range(0, linknumber)

        for i in noderange:
            data[0].append(get_activ())

        for l in linkrange:
            data[1].append([random.sample(noderange, k=1)[0],random.sample(noderange, k=1)[0],get_activ()])
        # d3js format: data = {"nodes": [{"id":1, "activity":70},{"id":2},{"id":3}], "links":[{"source":1,"target":2},{"source":2,"target":1, "activity":40}]}

        # data = [[0,get_activ(),2],[[0,1,get_activ()],[1,2,get_activ()]]]
        await websocket.send(json.dumps(data))
        # message = await websocket.send(json.dumps({"nodes": [{"id": x, "activity_state": 50}] , "links": [{"l": [1,x-1], "activity_state": 50}]}))
        time.sleep(1)



async def main():
    async with websockets.serve(client_connected_handler, "", 5678):
        await asyncio.Future()  # run forever

# def decorator(func):
#     def wrapper():
#         print("Something is happening before the function is called.")
#         func()
#         print("Something is happening after the function is called.")
#     return wrapper

if __name__ == "__main__":
    asyncio.run(main())



