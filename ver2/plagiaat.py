import serial
from math import cos, sqrt, atan2, floor, radians
values = [25, 25, 0, 120, 120, 120, 0, 170]

try:
    ser = serial.Serial('/dev/ttyACM1', 115200, timeout=1)
except:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

def write(move):
    print(move)
    ser.write(bytearray(move))
    #print("sent")
    while ser.inWaiting():
        print(ser.read())


while True:

    inp = input("sisesta:")
    if inp == "1":
        write(values)
    elif inp == "2":
        change = int(input("nr:"))
        if change == 0:
            data = int(input("väärtus:"))
            values[change] = data
        elif change == 1:
            data = int(input("väärtus:"))
            values[change] = data
        elif change == 2:
            data = int(input("väärtus:"))
            values[change] = data
        elif change == 3:
            data = int(input("väärtus:"))
            values[change] = data
        elif change == 4:
            data = int(input("väärtus:"))
            values[change] = data
        elif change == 5:
            data = int(input("väärtus:"))
            values[change] = data
        elif change == 6:
            data = int(input("väärtus:"))
            values[change] = data
        elif change == 7:
            data = int(input("väärtus:"))
            values[change] = data
    elif inp == "3":
        values[3] = 65
        values[4] = 65
        values[5] = 65
        write(values)
    else:
        break