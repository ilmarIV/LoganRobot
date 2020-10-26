from serial import Serial
import time

def serial_test():
    ser = Serial('COM5', baudrate=115200, timeout=1, write_timeout=1)

    if ser.isOpen():
        ser.close()
    ser.open()
    ser.isOpen()

    #ser.write(b'gs\n')

    ser.close()

