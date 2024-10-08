"""Cvv test."""

import asyncio

import json
import time
import websockets
import random
import numpy as np
import math


netsize = 20

zustand_t = 2 * np.random.random((netsize, 1)) - 1
zustand_t1 = np.copy(zustand_t)
neuro_aktivitaet = np.zeros(netsize)
durchgang = 0
shrinkingFaktor = 0.80

synapsenMatrix = np.multiply(
    (netsize * np.random.random((netsize, 3))).astype(int),
    [1, 1, (0.1 / netsize)]
)


input_mit_mapping = np.array([[0.5, 0], [1, 1], [0, 3], [0, 4]])
# output_mit_mapping = np.array([
#     [0.7, 7],
#     [0, 8],
#     [0.5, 9],
#     [0.1, 10],
#     [0.134, 11],
#     [0.321, 12]
# ])

output_mit_mapping = np.array([
    [0.321, 5],
    [0.9, 6],
])



async def client_connected_handler(websocket):
    """Client connected."""
    global zustand_t
    global zustand_t1
    global synapsenMatrix

    autorounds = 0
    while(True):
        data = [[],[]]
        train_data = {}
        train_data['input'] = np.ndarray(shape=(0,2))
        train_data['output'] = np.ndarray(shape=(0,2))

        if autorounds > 0:
            sel = 'll'
        else:
            sel = input(
                'press \n e (exit or and other for printing) \n'
                't for training \n a for add neuron \n'
                ' s for solution \n l for loop 100 \n'
                'll for loop 10000 \n na for neuro activity \n'
                'd to delete some synapse \n'
                'pz for print of state \n p for print of synapsenMatrix \n'
                'op for output \n ip for input \n'
            )

        if(sel == 'e'):
            exit()

        elif(sel == 'd'):
            removalLimit = 4
            for x in np.argwhere(abs(synapsenMatrix[:,2]) < 0.04):
                for i in x[:removalLimit]:
                    remove_synapse(i)
                    removalLimit -= 1

        elif(sel == 'll'):
            if autorounds == 0:
                autorounds = 1000  

            if((durchgang % 20) == 0):
                removalLimit = 5
                for a in range(removalLimit):
                    hits = np.argwhere(abs(neuro_aktivitaet) < 0.04)[:,0]
                    if hits.size > 0:
                        toremove = np.random.choice(hits, size=1)
                        for i in toremove:
                            remove_neuron(i)
                
            if((durchgang % 100) == 0):
                removalLimit = 4
                for x in np.argwhere(abs(synapsenMatrix[:,2]) < 0.01):
                    for i in x[:removalLimit]:
                        remove_synapse(i)
                        removalLimit -= 1
            if durchgang < 3000:
                if ((durchgang % 60) == 0):
                    add_neuron()
                    pass
                if((durchgang % 70) == 0):
                    x = np.argwhere(abs(neuro_aktivitaet) > 10)
                    if len(x) > 2:

                        add_synapse(
                            np.random.choice(x[:, 0]),
                            np.random.choice(x[:, 0]), 0.01
                        )
                        pass

            for inp in input_mit_mapping:
                zustand_t[int(inp[1])] = inp[0]

            step()
            autorounds -= 1
                

        elif(sel == 'l'):
            for i in range(100):
                for inp in input_mit_mapping:
                    zustand_t[int(inp[1])] = inp[0]
                step()

        elif(sel == 't'):
            for inp in input_mit_mapping:
                zustand_t[int(inp[1])] = inp[0]
            step()
            print('only output \n index \n {} \n loesung: \n {} \n'.format(
                output_mit_mapping[:, 1],
                zustand_t1[output_mit_mapping[:, 1].astype(int)]
            ))

        elif(sel == 'a'):
            print('adding neuron \n')
            add_neuron()

        elif(sel == 'p'):
            for d in synapsenMatrix:
                print(f"{int(d[0])},\t{int(d[1])},\t {d[2]}")
            # print(f"neues synapsenMatrix: \n {synapsenMatrix} \n")
            print("laenge synapsenMatrix: \n {} \n".format(len(synapsenMatrix)))

        elif(sel == 'pz'):
            print("zustand: \n {} \n".format(zustand_t1))
            print("laenge zustand: \n {} \n".format(len(zustand_t1)))

        elif(sel == 'op'):
            print("output: \n {} \n".format(output_mit_mapping))

        elif(sel == 'ip'):
            print("input: \n {} \n".format(input_mit_mapping))

        elif(sel == 'na'):
            print("neuro aktivitaet: \n {} \n".format(neuro_aktivitaet))
            print("laenge neuro aktivit.: \n {} \n".format(len(neuro_aktivitaet)))

        elif(sel == 's'):
            zustand_t1 = np.zeros((len(zustand_t), 1))
            for inp in input_mit_mapping:
                zustand_t[int(inp[1])] = inp[0]
            for row in synapsenMatrix:
                if row[1] not in input_mit_mapping[:, 1]:
                    zustand_t1[int(row[1])] += zustand_t[int(row[0])] * row[2]

            for ind, outOfSum in np.ndenumerate(zustand_t1):
                zustand_t1[ind] = 1 / (1 + np.exp(- outOfSum))

            print('only output \n index \n {} \n loesung: \n {} \n'.format(
                output_mit_mapping[:, 1],
                zustand_t1[output_mit_mapping[:, 1].astype(int)]
            ))

        else:
            print('noting selected \n \n')

        
        train_data['input'] = input_mit_mapping
        train_data['output'] = output_mit_mapping
                

        train_data['input'] = train_data['input'].tolist()
        train_data['output'] = train_data['output'].tolist()
        

        for i in neuro_aktivitaet:
            data[0].append(i)

        for l in synapsenMatrix:
            data[1].append([l[0],l[1],l[2]])

        await websocket.send(json.dumps(data))
        await websocket.send(json.dumps(train_data))


        # for row in synapsenMatrix:
        #     for inp in input_mit_mapping:
        #         zustand_t[inp[1]]=inp[0]
        #     zustand_t1[row[1]] = 1/(1+np.exp(-(zustand_t[row[1]] * row[2])))

        
        # await websocket.send(json.dumps(train_data))
    
# synapsenMatrix = np.array([
#     [0, 2, 0.2],
#     [0, 1, 0.1],
#     [0, 4, 0.1],
#     [0, 5, 0.1],
#     [0, 6, 0.1],
#     [1, 4, 0.5],
#     [1, 5, 0.5],
#     [1, 6, 0.5],
#     [1, 3, 0.5],
#     [1, 7, 0.5],
#     [1, 8, 0.5],
#     [1, 9, 0.5],
#     [2, 7, 0.5],
#     [2, 8, 0.5],
#     [2, 9, 0.5],
#     [3, 7, 0.5],
#     [3, 8, 0.5],
#     [3, 9, 0.5],
#     [4, 7, 0.5],
#     [4, 8, 0.5],
#     [4, 9, 0.5],
#     [5, 7, 0.5],
#     [5, 8, 0.5],
#     [5, 9, 0.5],
#     [6, 7, 0.5],
#     [6, 8, 0.5],
#     [6, 9, 0.5],
#     [7, 8, 0.5],
#     [7, 9, 0.5],
#     [2, 1, 0.5]
# ])

# input are nodes 0-4 of the network, output are nodes 7-12 of the nw
# first entry is the value, second entry is the noden


# print("synapsenMatrix: {}".format(synapsenMatrix))




def add_synapse(von, nach, gewicht=2 * np.random.random() - 1):
    """Addsyn."""
    global synapsenMatrix
    synapsenMatrix = np.append(synapsenMatrix, [[von, nach, gewicht]], axis=0)


def remove_synapse(d):
    """Delsyn."""
    global synapsenMatrix
    synapsenMatrix = np.delete(synapsenMatrix, d, axis=0)


def add_neuron():
    """Add neur."""
    global zustand_t
    global zustand_t1
    global netsize
    global neuro_aktivitaet
    last_index = len(zustand_t)
    zustand_t = np.append(zustand_t, [[0]], axis=0)
    zustand_t1 = np.copy(zustand_t)
    netsize += 1
    neuro_aktivitaet = np.zeros(netsize)
    for i in range(4):
        add_synapse(np.random.randint(last_index), last_index, 0.01)
        add_synapse(last_index, np.random.randint(last_index), 0.01)


def remove_neuron(ind):
    """Remove neur."""
    global zustand_t
    global zustand_t1
    global synapsenMatrix
    global netsize
    global neuro_aktivitaet
    if not any(c in output_mit_mapping[:, 1] for c in (ind, netsize - 1)):
        if not any(d in input_mit_mapping[:, 1] for d in (ind, netsize - 1)):
            neuro_aktivitaet = np.delete(neuro_aktivitaet, ind - 1, axis=0).copy()
            zustand_t = np.delete(zustand_t, ind - 1, axis=0).copy()
            zustand_t1 = np.copy(zustand_t)
            weg = []
            for key, x in enumerate(synapsenMatrix):
                if x[0] == ind or x[1] == ind:
                    if ind not in input_mit_mapping[:, 1]:
                        if ind not in output_mit_mapping[:, 1]:
                            weg.append([key])

            synapsenMatrix = np.delete(synapsenMatrix, weg, axis=0)
            for ikey, b in enumerate(output_mit_mapping[:, 1] > ind):
                if b:
                    output_mit_mapping[ikey, 1] -= 1

            for ikey, b in enumerate(input_mit_mapping[:, 1] > ind):
                if b:

                    input_mit_mapping[ikey, 1] -= 1

            for ikey, b in enumerate(synapsenMatrix[:, 0] > ind):
                if b:
                    synapsenMatrix[ikey, 0] -= 1

            for ikey, b in enumerate(synapsenMatrix[:, 1] > ind):
                if b:
                    synapsenMatrix[ikey, 1] -= 1
            netsize -= 1
            # neuro_aktivitaet = np.zeros(netsize)


def step():
    """Step func."""
    global zustand_t
    global zustand_t1
    global neuro_aktivitaet
    global netsize
    neuro_aktivitaet = np.zeros(netsize)
    global synapsenMatrix
    global durchgang
    zustand_t1 = np.zeros((len(zustand_t), 1))

    for row in synapsenMatrix:
        if row[1] not in input_mit_mapping[:, 1]:
            if row[1] > len(zustand_t1) - 1:
                # pdb.set_trace()
                pass
            # zustand_t1[row[1]] += 1 / (
            #     1 + np.exp(-(zustand_t[row[0]] * row[2]))
            # )
            zustand_t1[int(row[1])] += zustand_t[int(row[0])] * row[2]

    for ind, outOfSum in np.ndenumerate(zustand_t1):
        zustand_t1[ind] = 1 / (1 + np.exp(- outOfSum))

    zustand_tziel = np.copy(zustand_t1)
    tempdelta = np.copy(zustand_t1)

    for oup in output_mit_mapping:
        zustand_tziel[int(oup[1])] = oup[0]

    for laufindex in range(netsize):
        delta_zstd = (
            zustand_tziel[laufindex] -
            zustand_t1[laufindex]
        ) * (
            zustand_t[laufindex] *
            (1 - zustand_t[laufindex])
        )
        tempdelta[laufindex] = delta_zstd
        rows = np.where(synapsenMatrix[:, 1] == laufindex)
        for zeile in rows[0]:
            if len(zustand_t) <= synapsenMatrix[zeile][0]:
                # pdb.set_trace()
                pass
            synapsenMatrix[zeile][2] += (
                delta_zstd * zustand_t[int(synapsenMatrix[zeile][0])]
            ) * shrinkingFaktor
            neuro_aktivitaet[laufindex] += abs(synapsenMatrix[zeile][2])
    zustand_t = np.copy(zustand_t1) * shrinkingFaktor
    # synapsenMatrix = synapsenMatrix*shrinkingFaktor  
    durchgang += 1
    # print('error = tziel - t1  (und delta_zstd): \n')
    # # debugger.set_trace()
    # for (n, z) in enumerate(zustand_tziel):
    #     print('{:7.3f} = {:7.3f} - {:7.3f} ; {:7.3f}'.format(
    #         zustand_tziel[n][0] - zustand_t1[n][0], zustand_tziel[n][0],
    #         zustand_t1[n][0], tempdelta[n][0])
    #     )
    print('error {:7.3f} groesse {:3} lauf {:10}'.format(
        np.linalg.norm(
            zustand_tziel - zustand_t1), netsize, durchgang)
          )




async def main():
    async with websockets.serve(client_connected_handler, "", 5678):
        await asyncio.Future()



if __name__ == "__main__":
    asyncio.run(main())