import socket, threading
import json, copy, time
import pandas as pd
import numpy as np

class ZenoneComm:
    
    # set ip address to your address
    my_ip = '128.179.207.0'
    
    def __new__(cls, port):
        print("Creating a new instance of ZenoneComm.")
        return super().__new__(cls)
    
    def __init__(self, port):
        # Initialisation of the object
        self.my_port = port
        self.data      = dict()
        self.time2rcv  = None
        self.prev_data = {"P": [], "Q": []} 
        #self.new_data=None
        
    def recv_zenone(self):
        # method to read messages sent to (my_ip, my_port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.my_ip, self.my_port))
        print((self.my_ip, self.my_port))
        
        while True:

            timenow = time.time()
            p, addr = sock.recvfrom(65536)
            
            try:
                # store the previous data, might be useful
                self.prev_data["P"] += [copy.deepcopy(self.data['Data']['P'])]

            except KeyError:
                print("Nothing in data object")

            self.data = json.loads(p.decode('utf-8'))['Zenone'] # only keep data related to the Zenone PMU
            self.time2rcv = time.time()-timenow

    def get_power(self):
        """
        get the latest measured power
        """
        return {'P': self.data['Data']['P'], 'Q': self.data['Data']['Q']}
    
    def get_avg_power(self):
        
        sum = np.sum(np.array(self.prev_data['P']))
        sum += self.get_power()['P']
        
        avg = sum/(len(self.prev_data['P'])+1)
        self.prev_data['P'] = []
        return avg

    
    def get_timestamp(self):
        return self.data['Info']['Timestamp']

    def runner(self, dates, powers):

        # create and launch the UDP thread
        recv_scada_thread = threading.Thread(target=self.recv_zenone)
        recv_scada_thread.start()
        time.sleep(0.05)

        while True:
            sleep_t = 0.025
            t = time.time()
            try:
                # print(self.data)
                powers += [self.get_power()['P']]
                dates  += [self.get_timestamp()]
                print("Current measured power : {}".format(self.get_power()['P']))
                
            except KeyError:
                print("Data object might be empty -> no message has been received.")

            delta = time.time()-t
            time.sleep(sleep_t-min(delta, sleep_t))



def main():

    powers, dates = list(), list()
    
    try:
        comm = ZenoneComm(port=35210)
        comm.runner(dates, powers)
    finally:
        df = pd.DataFrame({'dates': dates, 'powers': powers})
        df.to_csv('zenone_power.csv')



if __name__=='__main__':
    main()