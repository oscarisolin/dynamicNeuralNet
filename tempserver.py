#!/usr/bin/env python

import asyncio

import json
import time
import websockets
import random
import numpy as np
import math

add_or_remove = [0, 1]

random.sample(add_or_remove, k=1)
amount = 50


def get_activ():
    return random.random()*100

async def client_connected_handler(websocket):
    print("Client connected")
    while True:
        data = [[],[]]
        train_data = {}
        train_data['input'] = np.ndarray(shape=(0,2))
        train_data['output'] = np.ndarray(shape=(0,2))

        

        nodenumber = random.sample(range(1, amount), k=1)[0]
        linknumber = random.sample(range(1, 2*amount), k=1)[0]
        input_number = random.sample(range(1, math.floor(nodenumber/3)+2), k=1)[0]
        output_number = random.sample(range(1, math.floor(nodenumber/3)+2), k=1)[0]
        noderange = range(0, nodenumber)
        linkrange = range(0, linknumber)

        for i in noderange:
            data[0].append(get_activ())

        for l in linkrange:
            data[1].append([random.sample(noderange, k=1)[0],random.sample(noderange, k=1)[0],get_activ()])
        # d3js format: data = {"nodes": [{"id":1, "activity":70},{"id":2},{"id":3}], "links":[{"source":1,"target":2},{"source":2,"target":1, "activity":40}]}

        for i in range(0, input_number):
            appendix = np.array([[get_activ(),random.sample(noderange, k=1)[0]]])
            train_data['input'] = np.append(train_data['input'], appendix, axis=0)
            pass
            # train_data['input'].append([get_activ(),random.sample(noderange, k=1)[0]])

        for o in range(0, output_number):            
            appendix = np.array([[get_activ(),random.sample(noderange, k=1)[0]]])
            train_data['output'] = np.append(train_data['output'], appendix, axis=0)
            # train_data['output'].append([get_activ(),random.sample(noderange, k=1)[0]])
        

        train_data['input'] = train_data['input'].tolist()
        train_data['output'] = train_data['output'].tolist()
        # train_data['input'] = np.array([
        #     [0.7, 7],
        #     [0, 8],
        #     [0.5, 9],
        #     [0.1, 10],
        #     [0.134, 11],
        #     [0.321, 12]
        # ])


        # train_data['output'] = np.array([
        #     [0.7, 7],
        #     [0, 8],
        #     [0.5, 9],
        #     [0.1, 10],
        #     [0.134, 11],
        #     [0.321, 12]
        # ])

        # output_mit_mapping = np.array([
        #     [0.7, 7],
        #     [0, 8],
        #     [0.5, 9],
        #     [0.1, 10],
        #     [0.134, 11],
        #     [0.321, 12]
        # ])


        # data = [[0,get_activ(),2],[[0,1,get_activ()],[1,2,get_activ()]]]
        await websocket.send(json.dumps(data))
        await websocket.send(json.dumps(train_data))
        
        time.sleep(3)



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



