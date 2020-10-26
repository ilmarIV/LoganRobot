import time

Kp = 0.6 * Ku
Ki = (1.2 * Ku) / Tu
Kd = (3 * Ku * Tu) / 40

output = 0
last_error = 0
integral = 0
seconds = time.time()

while True:
    dt = time.time() - seconds
    seconds = time.time()
    error = 320 - x
    integral = integral + error * dt
    deriative = (error - last_error) / dt
    output = int(Kp * error + Ki * integral + Kd * deriative)
    last_error = error

#output is the speed change for motors