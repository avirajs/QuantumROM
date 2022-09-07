from angle_QROM_optimized import get_angle_optim_QROM
import numpy as np
from scipy.linalg import hadamard


def get_mantissa_exp(int_arr, data_width):

    #convert from ints to binary vector for ease of conversion and remove trailing 0
    int_arr = np.array(int_arr).reshape(-1,1)
    bins = np.unpackbits(int_arr, axis=1)[:,-data_width:]

    #create vector when applied to binary matrix converts to floatingpoint
    bin_vect = np.power(np.ones(data_width)*2, np.arange(-1,-data_width-1,-1))

    #find exps by finding position of first 1
    exps = (bins!=0).argmax(axis=1)

    #create new mantissas by rolling the first 1 to the MSB
    for i in range(len(bins)):
        bins[i] = np.roll(bins[i],-exps[i])

    #convert mantissas to float
    float_mantissas = bins @ bin_vect


    return float_mantissas, exps

def get_angle_optim_QROM_phi(vals, hadamards = False,measurement=False, normalize=False):

    # create gray code permuted hadamard
    np.set_printoptions(suppress=True)

    def gray_hadamard(dim = 4 ):
        #hadamard matrix size must be power of 2, not odd
        n=np.int(np.ceil(np.log2(dim)))
        gray_ints = []
        for i in range(0, 1<<n):
            gray=i^(i>>1)
            gray_ints.append(gray)


        mat = hadamard(dim)[:,gray_ints]
        return mat

    def return_optim_control_idx(num_lines, control_idx, idxs = []):

        if(control_idx<0 or num_lines<control_idx):
            return idxs



        return_optim_control_idx(num_lines,control_idx-1, idxs)


        idxs.append(control_idx)
        # print(idxs)

        return_optim_control_idx(num_lines,control_idx-1, idxs)

        return idxs
        # return_optim_control_idx(num_lines,control_idx-1)


    curr_vector_size = len(vals)
    new_vector_size = int(2** np.ceil(np.log2(curr_vector_size)))
    #since we need to apply a hadamard which dimension size must be 2**n, where n is positive int, we need to pad our vector

    # print("og vector",vals)
    # print("value of pads",new_vector_size,curr_vector_size)
    vals = np.pad(vals,(0,new_vector_size-curr_vector_size))
    # print("Vector paded",vals)
    #normalization

    if normalize:
        vals = vals*2*np.pi/np.max(vals)


    M = gray_hadamard(new_vector_size)

    k = int(np.log2(new_vector_size))

    theta = 2**-k *(M.T @ vals)

    vals_size = len(vals)
    control_size = int(np.ceil(np.log2(vals_size)))

    # print("theta vectors",thetas)



    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

    addr_reg = QuantumRegister(control_size,"addr")
    data_reg = QuantumRegister(1,"data")
    ca = ClassicalRegister(control_size,"ca_m")
    cm = ClassicalRegister(1,"data_m")
    qc_min = QuantumCircuit(addr_reg,data_reg,cm,ca)

    if hadamards:
        qc_min.h(addr_reg[:])


    # qc_min.h(data_reg[0])


    #the CNOTs follow a rising and falling depths of an inorder binary search tree; more easily created using recursion
    cnot_controls = np.array(return_optim_control_idx(control_size,control_size-1))
    cnot_controls = cnot_controls.astype(np.uint8)
    cnot_controls = control_size-1 - cnot_controls

    # print("Theta size",theta.size)
    # print("Controls ",cnot_controls, cnot_controls.size)


    #CZ rotations are inbetween all CNOTS
    for idx,control in enumerate(cnot_controls):
        if theta[idx] != 0:
            qc_min.rz(theta[idx],data_reg[0])
        qc_min.cx(control,data_reg[0])


    # there is aditional theta and CNOT, that starts back over
    qc_min.rz(theta[-1],data_reg[0])


    qc_min.cx(0,data_reg[0])


    # qc_min.h(data_reg[0])

    if measurement:
        qc_min.measure(data_reg,cm)

        qc_min.measure(addr_reg,ca)


    return qc_min

def get_comb_results(qc_og, qc_min, measurement=True):

    import qiskit
    from qiskit.providers.aer import QasmSimulator
    from qiskit import QuantumCircuit, assemble, Aer
    from math import pi, sqrt
    from qiskit.visualization import plot_bloch_multivector, plot_histogram
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


    control_size = len(qc_og.qubits)-1
    addr_bits = QuantumRegister(control_size,"addr")
    data_bits = QuantumRegister(1,"data")

    ca = ClassicalRegister(control_size,"ca_m")
    cm = ClassicalRegister(1,"data_m")
    qc = QuantumCircuit(addr_bits,data_bits,cm,ca)

    # qc.h(data_bits[0])

    qc.compose(qc_og, inplace=True)


    qc.compose(qc_min, inplace=True)

    # qc.h(data_bits[0])

    if measurement:
        qc.measure(addr_bits,ca)

        qc.measure(data_bits,cm)



    return qc

def dynamic_range_QROM(int_arr,data_width,normalize=True):

    mantissa_thetas, exp_phis = get_mantissa_exp(int_arr, data_width)

    qc_min_thetas = get_angle_optim_QROM(mantissa_thetas,hadamards=False, measurement=False, normalize=normalize)
    qc_min_phis = get_angle_optim_QROM_phi(exp_phis, hadamards=False,measurement=False, normalize=normalize)


    qc_comb = get_comb_results(qc_min_phis, qc_min_thetas, measurement=False)

    return qc_comb
