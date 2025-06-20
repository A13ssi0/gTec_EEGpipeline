import socket
import numpy as np
import pickle
import io
import threading
import pygds 

HOST = '127.0.0.1'

class NautilusAcquisition:
    def __init__(self, port=12345, device=None, samplingRate=500, dataChunkSize=20):
        self.info = {
            'device': device,
            'samplingRate': samplingRate,
            'dataChunkSize': dataChunkSize,
            'channels':['FP1','FP2','F3','Fz','F4','T7','C3','Cz','C4','T8','P3','Pz','P4','PO7','PO8','Oz']
            }
        self.clients = []  # List of (conn, addr)
        self.clients_lock = threading.Lock()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = HOST
        self.port = port
        self.doClose = False

    def run(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f"[SERVER] Listening on {self.host}:{self.port}")
        threading.Thread(target=self._accept_clients, daemon=True).start()

        self.nautilus = pygds.GDS(gds_device=self.info['device']) 
        if self.info['device'] is None: self.info['device'] = self.nautilus.Name
        self.nautilus.SamplingRate = self.info['samplingRate']
        self.nautilus.SetConfiguration() 
        print("Starting acquisition...")
        self.nautilus.GetData(self.info['dataChunkSize'], more=self.data_callback)

        self.close()  # Close the server after starting acquisition

    def _accept_clients(self):
        while True:
            conn, addr = self.server_socket.accept()
            with self.clients_lock:
                self.clients.append((conn, addr))
            print(f"[+] Client connected: {addr}")
            threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()

    # def _handle_client(self, conn, addr):
    #     try:
    #         while True:
    #             # Clients just wait for data
    #             data = conn.recv(1)
    #             if not data:
    #                 break  # Client disconnected
    #     except Exception:
    #         pass
    #     finally:
    #         with self.clients_lock:
    #             self.clients = [(c, a) for (c, a) in self.clients if c != conn]
    #         print(f"[-] Client disconnected: {addr}")
    #         print(f"[SERVER] Remaining clients:",self.clients)
    #         if not self.clients:    # If no clients are connected, stop the acquisition
    #             print("[SERVER] No clients connected, stopping acquisition.")
    #             self.doClose = True
    #         conn.close()

    def _handle_client(self, conn, addr):
        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    break  # Client disconnected
                if data == b'GET_INFO':
                    info_pickled = pickle.dumps(self.info)
                    payload = len(info_pickled).to_bytes(4, 'big') + info_pickled
                    conn.sendall(payload)
        except Exception:
            pass
        finally:
            with self.clients_lock:
                self.clients = [(c, a) for (c, a) in self.clients if c != conn]
            print(f"[-] Client disconnected: {addr}")
            print(f"[SERVER] Remaining clients:", self.clients)
            if not self.clients:    # If no clients are connected, stop the acquisition
                print("[SERVER] No clients connected, stopping acquisition.")
                self.doClose = True
            conn.close()


    def data_callback(self, data):
        """Call this method when new data is ready from external device"""
        buffer = io.BytesIO()
        np.save(buffer, data)
        pickled = pickle.dumps(buffer.getvalue())
        payload = len(pickled).to_bytes(4, 'big') + pickled
        if self.doClose:
            print("[SERVER] Acquisition stopped due to no clients connected.")
            return False
        with self.clients_lock:
            disconnected = []
            for conn, addr in self.clients:
                try:
                    conn.sendall(payload)
                    #print(f"[SEND] Pushed to {addr}")
                except Exception as e:
                    print(f"[ERROR] Could not send to {addr}: {e}")
                    disconnected.append((conn, addr))
            # Remove disconnected clients
            self.clients = [c for c in self.clients if c not in disconnected]
        return True
    
    def close(self):
        """Call this method to stop the acquisition and close the server"""
        del self.nautilus     
        print("[SERVER] Acquisition stopped and server closed successfully.")
