from LoganRobot.proov import f, init
from threading import Thread

init()

Thread(name="f", target=f).start()
while True:
    print(a)
