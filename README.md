# Neutron Star Parameters as a Function of Density

This repository provides code that allows the calculation of relevant neutron star parameters as a function of density 
for a given set of Skyrme parameters t<sub>0</sub>, t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>, 
x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, &alpha;.

## Background Physics

To determine the composition of the nuclear matter inside the neutron star core, we follow 
[Chamel (2008)](https://academic.oup.com/mnras/article/388/2/737/977911)
and solve four equations related to baryon conservation, charge neutrality, beta equilibrium and muon production rate 
for a given set of Skyrme parameters. Implemented in `ns_eos/equilibrium_comp.py`, 
this provides number densities, particle fractions, Fermi wave numbers, 
Landau effective masses and dynamic effective masses related to entrainment 
(see [Chammel and Haensel (2006)](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.73.045802) 
for details) for a given baryon density.
To determine properties of the superfluids (London penetration depth and coherence lengths), 
we employ parametrisations for the singlet proton and triplet neutron gap introduced in 
[Andersson et al. (2005)](https://www.sciencedirect.com/science/article/abs/pii/S0375947405010572?via%3Dihub) 
(see also [Ho et al. (2012)](https://academic.oup.com/mnras/article/422/3/2632/1048899)). 
Details of the gap calculation can be found in `ns_eos/gap_parametrisation.py`.


## Getting Started

These instructions will provide you with a copy of the project and help you get it up and running on your local machine.
The repo contains an environment file that can be installed by running
```
$ conda env create -f environment.yaml
```
We recommend working within this environment when using the code. To install the package run
```
$ python setup.py install
```
To see if the installation has been successful, we provide an example file. Running 
```
$ python examples/generating_eos_data.py
```
will create three .txt files for three different sample equations of states (NRAPR, SLy4 and LNS) 
in `examples/data/`, providing a range of parameters as a function of baryon density.

Note that by default, an instance of the `EquationOfState` class will be created based on the Skyrme parameters 
of the NRAPR equation of state. If you would like to choose a different equation of state,
specify the corresponding Skyrme parameters t<sub>0</sub>, t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>, 
x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, &alpha; as shown in `examples/generating_eos_data.py`.

 
## Tests

All methods and functions in this repository have been tested. 
Tests can be found [here](https://github.com/vanessagraber/NS_EoS/tree/master/ns_eos/tests).
Running
```
$ python setup.py test
```
will run the tests and output a coverage report.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


