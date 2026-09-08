class PIDCon:
    Pgain = 0.0
    Igain = 0.0
    Dgain = 0.0

    Integral = 0
    maxI = 1.0
    lowI = -1.0

    def __init__(self, PGain, IGain, DGain, MaxI, LowI):
        self.Pgain = PGain
        self.Igain = IGain
        self.Dgain = DGain

        self.maxI = MaxI
        self.lowI = LowI
        self.prevError = 0

    def resetPID(self):
        self.Integral = 0

    def derivative(self, error, dt):
        dedt = ((error - self.prevError) / dt) * self.Dgain
        self.prevError = error
        return dedt

    def integral(self, error, dt):

        # prevent a glitch in the matrix
        if self.Igain == 0:
            return self.Integral

        self.Integral = (self.Integral + error * dt) * self.Igain
        if (self.Integral >= self.maxI):
            self.Integral = self.maxI
            return self.Integral
        elif (self.Integral <= self.lowI):
            self.Integral = self.lowI
            return self.Integral
        else:
            return self.Integral

    def proportional(self, error):
        return error * self.Pgain

    def control(self, target, current, dt):
        error = target - current
        proportional = self.proportional(error)
        derivative   = self.derivative(error, dt)
        integral     = self.integral(error, dt)
        return proportional + derivative + integral