import pyomo.environ as pyomo
import random
import matplotlib.pyplot as plt


# Create a model
model = pyomo.ConcreteModel()


# Time series
model.nt = pyomo.Param(initialize=20)
model.T = pyomo.Set(initialize=range(model.nt()))


# Create some random input data
xtmp = {}
random.seed(123)
for t in range(model.nt()):
    xtmp[t] = random.uniform(0, 100)


# Assign parameter values
model.X_in = pyomo.Param(model.T, initialize=xtmp)
model.XM = pyomo.Param(initialize=max(xtmp.values()))
model.X0 = pyomo.Param(initialize=50)





