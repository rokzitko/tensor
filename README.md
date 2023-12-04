# tensor
DMRG for SC-QD problem

This is an implementation of the density matrix renormalization group (DMRG) code for solving problems involving a superconducting (SC) bath described by the Richardson's pairing (picket-fence) Hamiltonian and a quantum dot (QD) described by the Anderson impurity model. This family of problems admit matrix-product-operator representations of the Hamiltonian that can be efficiently solved using the DMRG algorithm.

The code supports various extensions of the basic problem, including a charging term on the superconducting island (SI), capacitive coupling between the SI and the QD, magnetic field in arbitrary directions, and the spin-orbit coupling.

## Requirements

The code requires [iTensor](https://itensor.org/) version 3. 
A modern C++ compiler is necessary (tested with gcc 11.3).
The code makes use of HDF5 library (tested with hDF5 1.12.2), Boost
(tested with 1.79.0). Other dependencies come bundled with the code.

Use "make build" to build the code. Typical compilation time is under a minute.

The code is portable and should run on any unix-compatible operating
system. We mostly run it on CentOS 7.9 and Rocky 8.8.


## Input file format

The input file is a text file containing two blocks of parameters, one definining the problem and the main method parameters, the other defining the DMRG sweep procedure.

## Parameters

Here we provide brief descriptions for all parameters that can be defined in the input file.

### Physical parameters

* MPO - required parameter defines which MPO to use and thus defines the model. Some models do not have all parameters implemented! The most complete ones are MPO = Ec_V for the single channel model and MPO = 2ch_impFirst_V for the two channel model. The implemented MPOs are described below.
* N - number of sites.
* NBath - number of sites in the bath (either N or NBath have to be defined).
* band_level_shift - (true/false) whether to shift the bath level energies to recover a particle-hole symmetric model. Default = false!

* U - impurity level Coulomb repulsion.
* epsimp - impurity level energy.
* EZ_imp - impurity level Zeeman splitting.
* EZx_imp - magnetic field in the x direction on the impurity level. 
* alpha - superconducting pairing strength (has to be negative for a SC system).
* Ec - charging energy.
* n0 - optimal occupation of the superconductor. The term is Ec (n_SC - n0)^2.
* EZ_bulk - Zeeman splitting of all levels in the bath.
* EZx_bulk - magnetic field in the x direction for all levels in the bath.
* lambda - spin-orbit coupling in the bath.
* t - nearest neighbour hopping between levels in the bath.

* gamma - hybridization strength (defined by the level - impurity hopping parameter: gamma = pi/2 \* v^2). 
* V - capacitive coupling between the impurity and the bath. The term is V (n_SC - n0)\*(n_imp - nu).

#### Two channel model

When using the two channel MPOs, the bath parameters have to be specified for each bath separately. This can be done by appending the channel number (1 and 2) to the parameter name. Eg.: gamma1, gamma2, alpha1, alpha2, ...

Capacitive coupling is a special case:
* V1imp - capacitive coupling between the impurity and the first channel
* V2imp - capacitive coupling between the impurity and the second channel

#### Chain

When using the chain MPOs, it is possible to define parameters for each link in the chain separately by appending its number to the parameter name. The impurities, baths and hybridizations are enumerated separately. Assuming an alternating chain starting with a bath (SC_BathMPO_chain_alternating_SCFirst.h, only case implemented as of April 2022), the pairing strength in the third bath is alpha3, energy level of the second impurity is epsimp2 and the hybridization between the two is gamma4. 

It is also possible to create a homogeneous chain by specifying only a single set of parameters (as in the single channel case). These also act as default values when only a subset of link parameters are specified. 

#### Level specific parameters

Some bath parameters can be specified for each level. Pairing strength can be modulated on the nth level by giving eg. y_n = 0.4. This reduces the pairing of the n-th level by a factor of 0.4. All other levels will have the default value of pairing, given by alpha. 
Similarly, energy of each level can be specified by eps_n = xxx, and the impurity - level hopping by v_n = xxx.
This can be done for multi channel cases by specifying eg. eps1_n = xxx.  

#### Eta case

This is a special case of level specific parameter specification, where only a single level is different. Can only be used with some MPO implementations.

* eta - pairing strength modulation
* etasite - level for which to modulate the pairing strenght (default is at the Fermi level)

Specifying etasite = i and eta = x will produce the same Hamiltonian as specifying y_i = x. 

### Computation parameters

These parameters control the calculation. Their type is given in brackets. 

* solve_ndx (int) - optional parameter parsed directly from the command line. Defines which subproblems to solve. 
* stop_n (int) - optional parameter parsed directly from the command line. The calculation will stop after computing stop_n excited states.
* parallel (bool) - enable parallel computation in different subspaces.
* refisn0 (bool) - computation will center around the subspaces with n = nu + round(n0). For multiple superconducting islands, it will be centered around n = nu + round(n01) + round(n02) + ...
* sc_only (bool) - will initialize a state with an empty impurity level. Can be used in combination with gamma = 0 to completely decouple the impurity and only compute the bath model.

* spin1 (bool) - include Sz = 1 sectors for even n. 
* sz_override (bool) - if true, the sz ranges are controlled by sz_{even|odd}\_{min|max}.
* sz_even_max, sz_even_min, sz_odd_max, sz_odd_min (double) - control the maximal and minimal Sz values for odd and even n sectors. 

* flat_band (bool) - set half of the bath level energies to -flat_band_factor, and half of the levels to +flat_band_factor.
* flat_band_factor (float) - default = 0.
* band_rescale (float) - rescale the level energies of the band by the factor band_rescale. Note that this does not affect the rescaling of alpha (pairing strength) and lambda (SOC strength) parameters, thus the effect is different compared to changing the half-bandwidth D.

* reverse_second_channel_eps (bool) - reverse the bath energy levels in the second channel. In combination with MPOs with impurity in the middle creates an MPS symmetric to inversion of sites i <-> N-i, which is neccessary for calculating its parity.
 
* verbose (bool) - verbosity of the calculation. Will print various parameters during MPO creation and calculation.
* debug (bool) - enables debugging messages.

* enforce_total_spin (bool) - the total spin Sz(Sz + 1) is enforced when calculating eigenstates. Only works for the ground state (Jan 2023). Also not thoroughly tested.
This is done by minimizing <psi|H - w (S^2 - s^2)|psi> (instead of the standard <psi|H|psi>), where S^2 is the total spin MPO and s^2 = Sz(Sz + 1). w is the weight, set by spin_enforce_weight.
* spin_enforce_weight (float) - the prefactor in the expression above.

#### Sweep parameters

* nrH (int) - before starting the sweeping procedure, H is applied to the initial MPS nrH times. Seems to improve convergence. Default = 5.  
* nrsweeps (int) - number of DMRG sweeps to perform. If this is larger than the number of specified sweeps, the last one will be repeated untill nrsweeps sweeps are performed.
* Quiet, Silent - control the verbosity of the dmrg() function.
* EnergyErrgoal (float) - the sweeping will stop when the relative energy difference between consecutive sweeps is smaller than this value. Setting it to 0 will suppress the convergence output for each sweep.
* Weight (float) - energy cost of the overlap between GS and ES when calculating excited states. 

### State property calculation parameters

These parameters set what additional state properties are measured after obtaining the states.

* result_verbosity (int) - set which additional results are computed (quick way to enable various correlators). Default value is 0, setting to 1 enables calculation of impurity correlations, setting it to 2 enables calculations of correlation matrices and reduced density matrices on all sites.
* stdout_verbosity (int) - set which results are printed in standard output. All calculated results are always saved in the hdf5 file.

* excited_states (int) - compute n excited states. 
* excited_state (bool) - compute the first excited state.

* computeEntropy (bool) - von Neumann entropy at the bond between impurity and next site. Works as intended if p.impindex=1.
* computeEntropy_beforeAfter (bool) - von Neumann entropy at the bond before the impurity and after it. Works as intended if p.impindex!=1.

* measureAllDensityMatrices (bool) - computes local density matrix for each site. WARNING - for large systems this will blow up the output size (up to a few tens of Mb).
* chargeCorrelation (bool) - compute the impurity-bath correlation <n_imp n_i>. Imp-imp correlation is always given as the first value, irrespective of impindex! 
* spinCorrelation (bool) - compute the impurity-bath correlation <S_imp S_i>. NOTE: sum over i includes the impurity site! Imp-imp correlation is always given as the first value, irrespective of impindex! 
* pairCorrelation (bool) - compute the impurity-bath correlation <d d c_i^dag c_i^dag>. Imp-imp correlation is always given as the first value, irrespective of impindex! 
* hoppingExpectation (bool) - compute the hopping expectation value 1/sqrt(N) \sum_sigma \sum_i <d^dag c_i> + <c^dag_i d>. Imp-imp correlation is always given as the first value, irrespective of impindex! 

* spinCorrelationMatrix (bool) - compute the full correlation matrix <S_i S_j>.
* singleParticleDensityMatrix (bool) - compute the single-particle density matrix <cdag_i c_j>.

* overlaps (bool) - compute <i|j> overlap table in each subspace. Useful measure of convergence of excited states.
* cdag_overlaps (bool) - compute <i|cdag|j> overlap between subspaces which differ by 1 in total charge and 0.5 in total Sz.
* charge_susceptibility (bool) - compute <i|nimp|j> overlap table in each subspace.
* calcweights (bool) - calculates the spectral weights of the two closes spectroscopically availabe excitations.

* transition_dipole_moment (bool) - compute < i | nsc1 - ncs2 | j >. Two channel only.
* transition_quadrupole_moment (bool) - compute < i | nsc1 + ncs2 | j >. Two channel only. 
* measureChannelsEnergy (bool) - measure the energy gain for each channel separately. Uses special MPO implementations with H for one channel (MPO_2ch_channel1_only.h and MPO_2ch_channel2_only.h). Two channel with impurity level first only.
* measureParity (bool) - compute the channel parity. WARNING! This calculation is not thoroughly tested and might produce wrong and/or badly converged results! Will prolong the calculation for large systems. Automatically turns on the reverse_second_channel_eps parameter. Two channel only.


## MPOs implemented

The code implements a number of different Hamiltonians (in the form of their MPO representations). The model has to be specified with MPO = xxx. 

* std - single channel model, no charging energy! Impurity level first.
* Ec - single channel model with charging energy. Impurity level first.
* Ec_MF - version of the Ec MPO with mean field approximation of pairing. Impurity level first.
* Ec_V - single channel model with capacitive coupling V. Impurity level first.
* Ec_SO - single channel model with spin orbit coupling. Impurity level first, does not conserve Sz!
* Ec_V_SO - single channel model with spin orbit coupling and capacitive coupling. Impurity level first, does not conserve Sz!
* middle - single channel model with impurity level in the middle. 
* middle_Ec single channel model with charging energy and impurity level in the middle.
* t_SConly - only the bath Hamiltonian with nearest neighbour hopping between levels. Could potentially be used as a Hubbard model.
* Ec_t - single channel model with charging energy and nearest neighbour hopping in the bath. Impurity level first.

* middle_2channel - FOR TESTING PURPOSES ONLY. Two channel model with impurity between the channels, but the single channel Gamma as hybridization. 
* 2ch - two channel model with impurity in the middle. Charging energy included.
* 2ch_impFirst - two channel model with impurity on the first site. Performs better than 2ch for large values of U. 
* 2ch_impFirst_V - two channel model with capacitive coupling. Impurity level first.
* 2ch_t - two channel model with nearest neighbour hopping between bath levels. Impurity level first. 

* alternating_chain - model of a chain of alternating SC-QD-SC-QD- ...

MPOs implemented using the autoMPO functionality of ITensor. Useful for catching bugs in MPO implementation. Works well on small systems, too slow for large ones. 
* autoH - single channel model. Includes nearest neighbour hopping in the bath, but not charging energy.
* autoH_so - single channel model with spin orbit coupling.
* autoH_2ch - two channel model. Includes charging energy, capacitive coupling and nearest neighbour hopping terms.

## Output

The code produces textual output on the standard output, as well as more complete output to a binary HDF5 file.

## Examples (demo)

A number of examples is provided in the directory `test`. The expected
output files are included with the code. This can serve as a "known
answer" test of the code. The expected run time on a normal desktop
computer for all the tests is of the order of an hour.

## Known problems

Below are some special cases that are known to produce unhysical results or cause other problems.

### gamma = 0
Setting the hybridization to 0 will prohibit charge fluctuation from the impurity level to the bath. The calculated state will thus have the exact same impurity level as the initial state (ie. typically a single spin up). For a two channel case with the impurity level in the middle of the two channel, setting any gamma to 0 will prohibit charge recombination among the two channels.  
Correct results can be obtained by setting gamma to a small value, eg., 1e-8.

### Two channel imp middle with large U
In similar vein as the above, using large values of U with 2 channel MPOs that have the impurity level in the middle greatly increases the computation time. The cause is similiar, large U suppresses charge fluctuations between the two channels, slowing down the optimisation procedure. Somewhat better results are obtained by using MPOs with impurity level at the beggining. 



## Publications

This code has been used to produce the numerical results that formed the basis for the following publications:

* Subgap states in superconducting islands, 2021, https://doi.org/10.1103/PhysRevB.104.L241409
* Qubit based on spin-singlet Yu-Shiba-Rusinov states, 2022, https://doi.org/10.1103/PhysRevB.105.075129
* Excitations in a superconducting Coulombic energy gap, 2022, https://arxiv.org/abs/2101.10794

## Contact

The following people have contributed to this code:

Luka Pavešić
<mailto:luka.pavesic@ijs.si>  
Jozef Stefan Institute, Ljubljana, Slovenia

Daniel Bauernfeind
Center for Computational Quantum Physics, Simons Foundation Flatiron Institute, New York, New York 10010, USA

Rok Žitko  
<mailto:rok.zitko@ijs.si>  
<http://auger.ijs.si/nano>  
Jozef Stefan Institute, Ljubljana, Slovenia

The time-dependent variational principle code is a modified (compatibility fixes) version of the code from <https://github.com/ITensor/TDVP>.
