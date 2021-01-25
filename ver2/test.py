import time
from threading import Thread
from serial import Serial
import queue


def sendMainboard(ser, speeds_queue, stop_queue):
    while True:

        if not stop_queue.empty():
            return

        if not speeds_queue.empty():
            speeds = speeds_queue.get()

            speed_msg = "sd:" + str(speeds[0]) + ":" + str(speeds[1]) + ":" + str(speeds[2]) + "\r\n"
            throw_msg = "d:" + str(speeds[3]) + "\r\n"
            print(throw_msg)
            print(speed_msg)

            ser.write(speed_msg.encode('UTF-8'))
            time.sleep(0.02)

            ser.write(throw_msg.encode('UTF-8'))
            time.sleep(0.02)

            while ser.inWaiting() > 0:
                ser.read()


ser = Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
speeds_queue = queue.Queue()


stop_queue = queue.Queue()

mbt = Thread(target=sendMainboard, name="mbt", args=(ser, speeds_queue, stop_queue))

mbt.start()

while True:
    #time.sleep(0.39)
    speeds_queue.put([10,10,10,1200])
    #ser.write(b"d:900\r\n")
    time.sleep(1)

    #while ser.inWaiting() > 0:
        #ser.read()
    speeds_queue.put([10,0,0,1200])
    #ser.write(b"d:1200\r\n")
    time.sleep(1)

    #while ser.inWaiting() > 0:
        #ser.read()

