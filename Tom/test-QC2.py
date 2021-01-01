# general imports
import numpy as np
import math
import matplotlib.pyplot as plt
from braket.circuits import Circuit, circuit
from braket.devices import LocalSimulator
from braket.aws import AwsSession, AwsDevice
# local imports
from utils_circuit import get_unitary, adjoint
from utils_qaa import qaa

# monkey patch get_unitary() and adjoint() to the Circuit class
Circuit.get_unitary = get_unitary
Circuit.adjoint = adjoint
device = LocalSimulator()

# # or choose the cloud-based managed simulator to run your circuit
# device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

# Enter the S3 bucket you created during onboarding into the code that follows
my_bucket = f"amazon-braket-Your-Bucket-Name" # the name of the bucket
my_prefix = "Your-Folder-Name" # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)
@circuit.subroutine(register=True)
def minus_R_B(qubit):
    """
    Function to apply a minus sign to |B>|0>. This goal is achieved by applying XZX to the ancilla qubit.

    Args:
        qubit: the ancilla qubit on which we apply XZX.
    """
    # instantiate circuit object
    circ = Circuit()

    # Apply sequence XZX to given qubit
    circ.x(qubit).z(qubit).x(qubit)

    return circ

    # Helper function to apply rotation -R0
@circuit.subroutine(register=True)
def minus_R_zero(qubits, use_explicit_unitary=False):
    """
    Function to implement transformation: |0,0,...0> -> -|0,0,...0>, all others unchanged.

    Args:
        qubits: list of qubits on which to apply the gates
        use_explicit_unitary (default False): Flag to specify that we could instead implement
        the desired gate using a custom gate defined by the unitary diag(-1,1,...,1).
    """

    circ = Circuit()

    # If the use_explicit_matrix flag is True, we just apply the unitary defined by |0,0,...0> -> -|0,0,...0>
    if use_explicit_unitary:
        # Create the matrix diag(-1,1,1,...,1)
        unitary = np.eye(2**len(qubits))
        unitary[0][0]=-1
        # Add a gate defined by this matrix
        circ.unitary(matrix=unitary, targets=qubits)

    # Otherwise implement the unitary using ancilla qubits:
    else:
        # Flip all qubits. We now must check whether all qubits are |1>, rather than |0>.
        circ.x(qubits)

        # If we have only 1 qubit, we only must apply XZX to that qubit to pick up a minus sign on |0>
        if len(qubits) < 2:
            circ.z(qubits)

        # For more qubits, we use Toffoli (or CCNOT) gates to verify the qubits are in |1> (after applying X)
        else:

            # Dynamically add ancilla qubits, starting on the next unused qubit in the circuit
            # NOTE: if this subroutine is being applied to a subset of qubits in a circuit, these ancilla
            # registers may already be used. We could pass in circ as an argument and add ancillas outside of
            # circ.targets instead, if desired.
            ancilla_start = max(qubits) + 1

            # Check that the first two register qubits are both 1's using a CCNOT on a new ancilla qubit.
            circ.ccnot(qubits[0],qubits[1],ancilla_start)

            # Now add a CCNOT from each of the next register qubits, comparing with the ancilla we just added.
            # Target on a new ancilla. If len(qubits) is 2, this does not execute.
            for ii,qubit in enumerate(qubits[2:]):
                circ.ccnot(qubit,ancilla_start+ii, ancilla_start+ii+1)

            # A Z gate applied to the last ancilla qubit gives a minus sign if all register qubits are |1>
            ancilla_end = ancilla_start + len(qubits[2:])
            circ.z(ancilla_end)

            # Now uncompute to disentangle the ancilla qubits by applying CCNOTs in the reverse order to the previous.
            for jj,qubit in enumerate(reversed(qubits[2:])):
                circ.ccnot(qubit,ancilla_end-jj-1, ancilla_end-jj)

            # Finally undo the last CCNOT on the first two register qubits.
            circ.ccnot(qubits[0],qubits[1],ancilla_start)

        # Flip all qubits back
        circ.x(qubits)

    return circ

@circuit.subroutine(register=True)
def grover_iterator(A,flag_qubit,qubits=None,use_explicit_unitary=False):
    """
    Function to implement the Grover iterator Q=A R_0 A* R_B.

    Args:
        A: Circuit defining the unitary A
        flag_qubit: Specifies which of the qubits A acts on labels the good/bad subspace.
                    Must be an element of qubits (if passed) or A.qubits.
        qubits: list of qubits on which to apply the gates (including the flag_qubit).
                If qubits is different from A.qubits, A is applied to qubits instead.
        use_explicit_unitary: Flag to specify that we should implement R_0 using using a custom
                              gate defined by the unitary diag(-1,1,...,1). Default is False.
    """
    # If no qubits are passed, apply the gates to the targets of A
    if qubits is None:
        qubits = A.qubits
    else:
        # If qubits are passed, make sure it's the right number to remap from A.
        if len(qubits)!=len(A.qubits):
            raise ValueError('Number of desired target qubits differs from number of targets in A'.format(flag_qubit=repr(flag_qubit)))

   # Verify that flag_qubit is one of the qubits on which A acts, or one of the user defined qubits
    if flag_qubit not in qubits:
        raise ValueError('flag_qubit {flag_qubit} is not in targets of A'.format(flag_qubit=repr(flag_qubit)))

    # Instantiate the circuit
    circ = Circuit()

    # Apply -R_B to the flag qubit
    circ.minus_R_B(flag_qubit)

    # Apply A^\dagger. Use target mapping if different qubits are specified
    circ.add_circuit(A.adjoint(),target=qubits)

    # Apply -R_0
    circ.minus_R_zero(qubits,use_explicit_unitary)

    # Apply A, mapping targets if desired.
    circ.add_circuit(A,target=qubits)

    return circ

@circuit.subroutine(register=True)
def qaa(A,flag_qubit,num_iterations,qubits=None,use_explicit_unitary=False):
    """
    Function to implement the Quantum Amplitude Amplification Q^m, where Q=A R_0 A* R_B, m=num_iterations.

    Args:
        A: Circuit defining the unitary A
        flag_qubit: Specifies which of the qubits A acts on labels the good/bad subspace.
                    Must be an element of qubits (if passed) or A.qubits.
        num_iterations: number of applications of the Grover iterator Q.
        qubits: list of qubits on which to apply the gates (including the flag_qubit).
                If qubits is different from A.qubits, A is applied to qubits instead.
        use_explicit_unitary: Flag to specify that we should implement R_0 using using a custom
                              gate defined by the unitary diag(-1,1,...,1). Default is False.
    """
    # Instantiate the circuit
    circ = Circuit()

    # Apply the Grover iterator num_iterations times:
    for _ in range(num_iterations):
        circ.grover_iterator(A,flag_qubit,qubits,use_explicit_unitary)

    return circ


#MAIN CODE


    # Set up the state A|00>
flag_qubit = 1
epsilon = 0.05
A_circ = Circuit().ry(0, epsilon).cnot(0,1)
print (A_circ)

# Add marginal probability for flag qubit as result Type
A_circ.probability(target=[flag_qubit])

# Let's find the probability of measuring |11> for different values of m, the number of applications of QAA:
probabilities = []
stepsize = 2
iterations = range(1, 40, stepsize)
for m in iterations:

    # Get circuit object
    circ = Circuit()
    # Apply QAA using A defined by A_circ
    circ.qaa(A_circ, flag_qubit, m, use_explicit_unitary=True)

    # Classically simulate the circuit
    # Give the correct device.run call depending on whether the device is local or cloud-based (managed)
    if device.name == 'DefaultSimulator':
        task = device.run(circ, shots=000)
    else:
        task = device.run(circ, s3_folder, shots=1000, poll_timeout_seconds=60*60)

    # Get result
    result = task.result()
    # Append the probability of measuring |11> for this value of m.
    probabilities.append(result.values[0][1])

# Get analytical result for comparison
probs_theo = [np.sin((2*mm+1)*epsilon/2)**2 for mm in iterations]

print ("pippo")
print (iterations)
print(probabilities)
# Plot the results
plt.figure(figsize=(7,5))
plt.plot(iterations, probabilities, 'o');
plt.plot(iterations, probs_theo);
plt.xlabel('Number of Iterations');
plt.ylabel('Probability of measuring flag qubit in |1>');
