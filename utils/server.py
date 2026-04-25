import socket
import ast
import numpy as np
from scipy.signal import welch

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
    Online PSD processor.

    Input:
        TCP stream of EEG chunks shaped:
            (samples, channels)

    Output:
        Flattened PSD vector ordered as:
            freq1[ch1..chN], freq2[ch1..chN], ...

    """

    def __init__(
        self,
        managerPort=25798,
        host="127.0.0.1",
        win_length=1.0,
        update_rate=0.25,
        freq_limit=50,
        nperseg=256
    ):
        self.host = host
        self.name = "Filter"

        # PSD parameters
        self.win_length = win_length
        self.update_rate = update_rate
        self.freq_limit = freq_limit
        self.nperseg = nperseg

        # runtime vars
        self.sfreq = None
        self.buffer = None
        self.buffer_samples = None
        self.update_samples = None
        self.samples_since_update = 0
        self.freq_mask = None
        self.freqs = None

        neededPorts = [
            "InfoDictionary",
            "EEGData",
            "FilteredData",
            "host"
        ]

        self.init_sockets(managerPort, neededPorts)

    # =====================================================
    # SOCKET INIT
    # =====================================================

    def init_sockets(self, managerPort, neededPorts):

        portDict = get_serversPort(
            host=self.host,
            managerPort=managerPort,
            neededPorts=neededPorts
        )

        if portDict["host"] is not None:
            self.host = portDict["host"]

        self.EEGPort = portDict["EEGData"]
        self.InfoDictPort = portDict["InfoDictionary"]

        self.Filtered_socket = TCPServer(
            host=self.host,
            port=portDict["FilteredData"],
            serverName=self.name,
            node=self
        )

    # =====================================================
    # INFO
    # =====================================================

    def request_info(self):

        wait_for_udp_server(self.host, self.InfoDictPort)

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:

            send_udp(sock, (self.host, self.InfoDictPort), "GET_INFO")
            _, raw_info, _ = recv_udp(sock)

        try:
            info = ast.literal_eval(raw_info)
        except Exception as e:
            print(f"[{self.name}] Info parse error: {e}")
            info = {}

        self.sfreq = info.get("SampleRate", 512)

        self.buffer_samples = int(self.win_length * self.sfreq)
        self.update_samples = int(self.update_rate * self.sfreq)

        print(f"[{self.name}] SampleRate = {self.sfreq}")
        print(f"[{self.name}] Window = {self.buffer_samples} samples")
        print(f"[{self.name}] Update = {self.update_samples} samples")

    # =====================================================
    # BUFFER
    # =====================================================

    def init_buffer(self, n_channels):

        self.buffer = np.zeros(
            (self.buffer_samples, n_channels),
            dtype=np.float32
        )

    def update_buffer(self, chunk):

        n = chunk.shape[0]

        if n >= self.buffer_samples:
            self.buffer[:] = chunk[-self.buffer_samples:]

        else:
            self.buffer[:-n] = self.buffer[n:]
            self.buffer[-n:] = chunk

        self.samples_since_update += n

    # =====================================================
    # PSD
    # =====================================================

    def compute_psd(self):

        f, pxx = welch(
            self.buffer,
            fs=self.sfreq,
            nperseg=min(self.nperseg, self.buffer.shape[0]),
            axis=0
        )

        if self.freq_mask is None:
            self.freq_mask = f <= self.freq_limit
            self.freqs = f[self.freq_mask]

        pxx = pxx[self.freq_mask]       # (freqs, channels)

        # log power
        pxx = np.log10(pxx + 1e-12)

        return pxx

    def format_output(self, psd):

        # psd shape = (freqs, channels)
        # flatten row-wise:
        # freq1[ch1..N], freq2[ch1..N]

        return psd.reshape(-1).astype(np.float32)

    # =====================================================
    # MAIN LOOP
    # =====================================================

    def run(self):

        self.request_info()

        self.Filtered_socket.start()

        tcp_sock = None

        try:
            tcp_sock = wait_for_tcp_server(self.host, self.EEGPort)

            print(f"[{self.name}] Connected to EEG source")

            while not self.Filtered_socket._stopEvent.is_set():

                try:
                    _, chunk = recv_tcp(tcp_sock)

                    if chunk is None:
                        continue

                    if self.buffer is None:
                        self.init_buffer(chunk.shape[1])

                    self.update_buffer(chunk)

                    if self.samples_since_update >= self.update_samples:

                        self.samples_since_update = 0

                        psd = self.compute_psd()

                        payload = self.format_output(psd)

                        self.Filtered_socket.broadcast(payload)

                except Exception as e:
                    print(f"[{self.name}] Processing error: {e}")
                    break

        except Exception as e:
            print(f"[{self.name}] Connection error: {e}")

        finally:

            if tcp_sock is not None:
                try:
                    tcp_sock.close()
                except:
                    pass

            self.close()

    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        safeClose_socket(self.Filtered_socket, name=self.name)

    def __del__(self):

        try:
            self.close()
        except:
            pass