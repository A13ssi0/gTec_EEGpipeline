import socket
import ast
import numpy as np
from scipy.signal import welch

from py_utils.buffer import Buffer

from utils.server import (
    TCPServer,
    recv_udp,
    recv_tcp,
    wait_for_udp_server,
    wait_for_tcp_server,
    send_udp,
    safeClose_socket,
    get_serversPort
)


class Filter:
    """
    Online PSD filter.

    INPUT:
        TCP stream of EEG chunks shaped:
            (samples, channels)

    OUTPUT:
        Flattened PSD vector ordered by frequency:

        freq1[ch1..chN],
        freq2[ch1..chN],
        freq3[ch1..chN], ...
    """

    def __init__(self, managerPort=25798, host='127.0.0.1'):
        self.name = "Filter"
        self.host = host

        # PSD settings
        self.win_length = 1.0      # seconds
        self.update_rate = 0.25    # seconds
        self.freq_limit = 50.0     # Hz

        # runtime
        self.info = {}
        self.sfreq = None

        self.buffer = None
        self.buffer_samples = None
        self.update_samples = None
        self.samples_since_update = 0

        self.freq_mask = None
        self.freqs = None

        neededPorts = [
            'InfoDictionary',
            'EEGData',
            'FilteredData',
            'host'
        ]

        self.init_sockets(managerPort, neededPorts)

    # ---------------------------------------------------
    # SOCKETS
    # ---------------------------------------------------
    def init_sockets(self, managerPort, neededPorts):
        portDict = get_serversPort(
            host=self.host,
            managerPort=managerPort,
            neededPorts=neededPorts
        )

        if portDict['host'] is not None:
            self.host = portDict['host']

        self.InfoDictPort = portDict['InfoDictionary']
        self.EEGPort = portDict['EEGData']

        self.Filtered_socket = TCPServer(
            host=self.host,
            port=portDict['FilteredData'],
            serverName=self.name,
            node=self
        )

    # ---------------------------------------------------
    # LOAD INFO
    # ---------------------------------------------------
    def load_info(self):
        wait_for_udp_server(self.host, self.InfoDictPort)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            send_udp(udp_sock, (self.host, self.InfoDictPort), "GET_INFO")
            _, raw_info, _ = recv_udp(udp_sock)

        try:
            self.info = ast.literal_eval(raw_info)
        except Exception as e:
            print(f"[{self.name}] Could not parse info: {e}")
            self.info = {}

        self.sfreq = int(self.info.get("SampleRate", 250))

        print(f"[{self.name}] SampleRate = {self.sfreq}")

    # ---------------------------------------------------
    # PSD
    # ---------------------------------------------------
    def compute_psd(self, data):
        """
        Input:
            data = (samples, channels)

        Output:
            (channels, freqs)
        """
        nperseg = min(256, data.shape[0])

        freqs, pxx = welch(
            data,
            fs=self.sfreq,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            axis=0
        )

        if self.freq_mask is None:
            self.freq_mask = freqs <= self.freq_limit
            self.freqs = freqs[self.freq_mask]

        pxx = pxx[self.freq_mask]      # (freqs, channels)

        # log transform
        pxx = np.log10(pxx + 1e-12)

        return pxx.T                  # (channels, freqs)

    def flatten_psd(self, psd):
        """
        Convert:

        (channels, freqs)

        to:

        freq1[ch1..N], freq2[ch1..N], ...
        """
        return (
            psd.T
               .reshape(-1)
               .astype(np.float32)
        )

    # ---------------------------------------------------
    # RUN
    # ---------------------------------------------------
    def run(self):
        self.load_info()

        self.Filtered_socket.start()

        tcp_sock = None

        try:
            tcp_sock = wait_for_tcp_server(self.host, self.EEGPort)

            print(f"[{self.name}] Connected to EEG stream")

            while not self.Filtered_socket._stopEvent.is_set():

                _, chunk = recv_tcp(tcp_sock)
                chunk = np.asarray(chunk, dtype=np.float32)

                if chunk.ndim != 2:
                    continue

                # first chunk -> init buffer
                if self.buffer is None:
                    n_channels = chunk.shape[1]

                    self.buffer_samples = int(
                        self.win_length * self.sfreq
                    )

                    self.update_samples = int(
                        self.update_rate * self.sfreq
                    )

                    self.buffer = Buffer(
                        (self.buffer_samples, n_channels)
                    )

                # update rolling buffer
                self.buffer.add_data(chunk)

                # wait until full buffer
                if not self.buffer.isFull:
                    continue

                self.samples_since_update += chunk.shape[0]

                # exact timing (no drift)
                while self.samples_since_update >= self.update_samples:
                    self.samples_since_update -= self.update_samples

                    data = self.buffer.get_data()

                    psd = self.compute_psd(data)

                    packet = self.flatten_psd(psd)

                    self.Filtered_socket.broadcast(packet)

        except Exception as e:
            print(f"[{self.name}] Error: {e}")

        finally:
            if tcp_sock is not None:
                try:
                    tcp_sock.close()
                except:
                    pass

            self.close()

    # ---------------------------------------------------
    # CLOSE
    # ---------------------------------------------------
    def close(self):
        safeClose_socket(self.Filtered_socket, name=self.name)

    def __del__(self):
        try:
            self.close()
        except:
            pass