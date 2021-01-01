# AWS import Boto3
import boto3
# AWS imports: Import Braket SDK modules
from braket.circuits import Circuit
from braket.aws import AwsDevice
# OS import to load the region to use
import os
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
# The region name must be configured


# Create the Teleportation Circuit
circ = Circuit()
# Put the qubit to teleport in some superposition state, very simple
# in this example
circ.h(0)
# Create the entangled state (qubit 1 reamins in Alice while qubit 2
# is sent to Bob)
circ.h(1).cnot(1, 2)
# Teleportation algorithm
circ.cnot(0, 1).h(0)
# Do the trick with deferred measurement
circ.h(2).cnot(0, 2).h(2)  # Control Z 0 -> 2 (developed because
                           # IonQ is not having native Ctrl-Z)
circ.cnot(1, 2)            # Control X 1 -> 2

print(circ)
# Run circuit

from braket.devices import LocalSimulator


device = LocalSimulator()

result = device.run(circ,  shots=1000)
counts = result.result().measurement_counts
print(counts)    
