from serial import Serial

ser = Serial('/dev/ttyACM0', baudrate=115200, timeout=1)

def mainboardMSG(speed_1, speed_2, speed_3):
    msg = "sd:" + str(speed_1) + ":" + str(speed_2) + ":" + str(speed_3) + "\r\n"
    ser.write(msg.encode('UTF-8'))
    while ser.inWaiting() > 0:
        ser.read()
