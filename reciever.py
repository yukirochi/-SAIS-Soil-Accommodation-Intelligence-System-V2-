"""FIFO receiver for four independent USB-RS485 serial connections.

Each node sends one soil-moisture percentage followed by a newline.
Install the serial dependency with: python -m pip install pyserial
"""

from queue import Queue
from threading import Thread

import serial

from predict import clean_input, predict

PORTS = {
    "node_1": "COM3",
    "node_2": "COM4",
    "node_3": "COM5",
    "node_4": "COM6",
}
BAUD_RATE = 9600
SOIL_MOISTURE_THRESHOLD = 30.0
TEMPERATURE_C = 30.0
HUMIDITY_PERCENT = 70.0

requests = Queue()


def read_node(node_id, port):
    """Read requests from one node and add them to the shared FIFO."""
    with serial.Serial(port, BAUD_RATE, timeout=None) as connection:
        print(f"{node_id} connected on {port}")
        while True:
            line = connection.readline().decode("utf-8").strip()
            if line:
                requests.put((node_id, connection, line))


def process_request(node_id, connection, line):
    """Predict one soil reading and send the result to its node."""
    try:
        soil_moisture = float(line)
        if not 0 <= soil_moisture <= 100:
            raise ValueError("soil moisture must be between 0 and 100 percent")
        deficit = max(SOIL_MOISTURE_THRESHOLD - soil_moisture, 0.0)
        inputs = clean_input(deficit, TEMPERATURE_C, HUMIDITY_PERCENT)
        water = predict(inputs)
        connection.write(f"{water:.3f}\n".encode("utf-8"))
        print(f"{node_id}: moisture={soil_moisture:.2f}% -> {water:.3f} L/m^2")
    except ValueError as error:
        connection.write(f"ERROR: {error}\n".encode("utf-8"))
        print(f"{node_id}: invalid request {line!r}")


def process_requests():
    """Predict queued requests in arrival order and reply to their node."""
    while True:
        request = requests.get()
        try:
            process_request(*request)
        finally:
            requests.task_done()


if __name__ == "__main__":
    for node_id, port in PORTS.items():
        Thread(target=read_node, args=(node_id, port), daemon=True).start()
    print("Waiting for node requests. Press Ctrl+C to stop.")
    process_requests()
