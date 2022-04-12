# tensor
DMRG for SC-QD problem

This is an implementation of the density matrix renormalization group (DMRG) code for solving problems involving a superconducting (SC) bath described by the Richardson's pairing (picket-fence) Hamiltonian and a quantum dot (QD) described by the Anderson impurity model. This family of problems admit matrix-product-operator representations of the Hamiltonian that can be efficiently solved using the DMRG algorithm.

The code supports various extensions of the basic problem, including a charging term on the superconducting island (SI), capacitive coupling between the SI and the QD, magnetic field in arbitrary directions, and the spin-orbit coupling.

## Requirements

The code requires iTensor. A modern C++ compiler is necessary.

## Input file format

The input file is a text file containing two blocks of parameters, one definining the problem and the main method parameters, the other defining the DMRG sweep procedure.

## Parameters

Here we provide brief descriptions for all parameters that can be defined in the input file.

## Output

The code produces textual output on the standard output, as well as more complete output to a binary HDF5 file.

## Examples

A number of examples is provided in the directory `testing`.

## Publications

This code has been used to produce the numerical results that formed the basis for the following publications:

a
b
c

## Contact

The following people have contributed to this code:

Luka

Daniel

Rok
rok.zitko@ijs.si
http://auger.ijs.si/nano
Jozef Stefan Institute, Ljubljana, Slovenia
