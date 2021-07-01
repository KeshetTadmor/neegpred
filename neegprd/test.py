class Time:
    def __init__(self, hour=0, minute=0, second=0):
        self.hour = hour
        self.minute = minute
        self.second = second
        self.time_in_seconds = self.hour*3600 + self.minute*60 + self.second
        if self.time_in_seconds > 86399:  # the number of seconds in 23:59:59
            self.time_in_seconds = 0
        self.hour = int(
            (self.time_in_seconds - self.time_in_seconds % 3600) / 3600)
        self.second = int(
            (self.time_in_seconds - self.hour*3600) % 60)
        self.minute = int(
            (self.time_in_seconds-self.hour*3600 - self.second)/60)

    def __str__(self):
        print(str(self.hour)+':'+str(self.minute)+':'+str(self.second))

    def is_after(self, time_point):
        if self.time_in_seconds > time_point.time_in_seconds:
            return True
        else:
            return False


t1 = Time(8, 23, 0)
t2 = Time(7, 20, 40)
t1.__str__()
print(t1.is_after(t2))

#print(t1.hour, t1.minute, t1.second)
#print(23*3600 + 59 * 60 + 59)
