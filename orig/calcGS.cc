#include "itensor/all.h" 
#include "SC_BathMPO.h"
#include "lanczos.h"

#include <iomanip>
#include <vector>

using namespace itensor;

void GetBathParams(double U, double gamma, std::vector<double>& eps, std::vector<double>& V, int NBath);
void MyDMRG(MPS& psi, MPO& H, double& energy, Args args);
void FindGS(MPO& H, const SiteSet& sites, std::string inputfn,  Args args);
void MeasureOcc(MPS& psi, const SiteSet& sites);



int main(int argc, char* argv[]){ 
    if(argc!=2){
        std::cout<<"Please provide input file. Usuage ./calcGS inputfile.txt"<<std::endl;
        return 0;
    }

    //get input parameters using inputgroup from itensor
    auto input = InputGroup(argv[1], "params");
    //number of sites
    int N = input.getInt("N"), NBath = N-1;
    double U=input.getReal("U"), Vbar=input.getReal("Vbar"), gamma =input.getReal("gamma");


    //sites is an ITensor thing. it defines the local hilbert space and
    //operators living on each site of the lattice
    //for example sites.op("N",1) gives the pariticle number operator 
    //on the first site
    auto sites = Hubbard(N);

    //args is used to store and transport parameters between various functions
    Args args;

    std::vector<double> eps; // vector containing on-site energies of the bath !one-indexed! 
    std::vector<double> V, noV(N+1); //vector containing hoppinga amplitudes to the bath !one-indexed! 
    
    GetBathParams(U,gamma, eps, V, NBath);

    //MPO is the hamiltonian in "MPS-form" after this line it is still a trivial operator
    MPO H(sites);
    //defined in SC_BathMPO.h, fills the MPO with the necessary entries
    Fill_SCBath_MPO(H, sites, eps, V, U, Vbar/NBath );

    //calculates the ground state of the three relevant particle number sectors
    FindGS(H, sites , argv[1], args);
}



//calculates the groundstates and the energie of the three relevant particle number sectors
void FindGS(MPO& H, const SiteSet& sites, std::string inputfn, Args args){


    auto N = length(H);
    std::vector<double> energies(0);
    std::vector<int> numPart(0);

    //loop over the three sectors of particle number we are interested in 
    //i=1: half filling
    //i=2: half filling + 1
    //i=3: half filling - 1
    for(auto i: range1(3)){
        auto input = InputGroup(inputfn,"sweeps");
        auto sw_table = InputGroup(input,"sweeps");

        //the sweeps object defines the accuracy used for each update cycle in DMRG
        //the used parameters are read from the input file with the following meaning:
        // maxdim:  maximal bond dimension -> start low approach ground state roughly, then increase
        // mindim:  minimal bond dimension used, can be helpfull to improve DMRG convergence
        // cutoff:  truncated weight i.e.: sum of all discared squared Schmidt values 
        // niter:   during DMRG, one has to solve an eigenvalue problem usually using a krylov 
        //          method. niter is the number of krylov vectors used. Since DMRG is variational
        //          it does not matter if we use only very few krylov vectors. We only want to 
        //          move in the direction of the ground state.
        // noise:   if non-zero a so-called noise term is added to DMRG which improves the convergence.
        //          this term must be zero towards the end, as the ground state is otherwise incorrect.
        auto sweeps = Sweeps(15,sw_table);
        //doing 15 sweeps here


        //initialize the MPS in a product state with the correct particle number
        //since the itensor conserves the number of particles, in this step we 
        //choose in which sector of particle number we perform the calculation
        auto state = InitState(sites);
        int totPart = 0;
        for(auto i : range1( N/2 ) ){ 
            state.set(i,"UpDn");
            totPart +=2;
        }
        if(i==2){
            //add up particle to next site
            state.set(N/2+1,"Up");
            totPart += 1;
        }
        if(i==3)
        {
            //remove up particle from last site
            state.set(N/2,"Dn");
            totPart -= 1;
        }
        numPart.push_back(totPart);
        std::cout <<"Using sector with " << totPart << " number of Particles"<<std::endl;

        MPS psi(state);
        
        //apply the MPO a couple of times to get DMRG started. otherwise it might not converge
        for(auto i : range1(5)){
            psi = applyMPO(H,psi,args);
            psi.noPrime().normalize();
        }




        //call itensor dmrg another possibility is to use the function MyDMRG
        //written by me which uses my own lanczos towards the end of DMRG:

        //double energy;
        //args.add("Cutoff",1E-12);
        //args.add("maxsweeps",10);
        //MyDMRG(psi, H, energy, args);
        auto [energy, GS] = dmrg(H,psi,sweeps,{"Quiet",true});

        
        printfln("Ground state energy = %.20f",energy);
        MeasureOcc(GS, sites);
        energies.push_back(energy);
    } 


    //print energies and sectors
    for( auto i : range( energies.size() ) ){
        std::cout<< "n = "<< numPart[i] << "  E= "<< std::setprecision(16)<< energies.at(i) << std::endl; 
    }

}

//prints the occupation number of an MPS psi
//instructive to learn how to calculate local observables
void MeasureOcc(MPS& psi, const SiteSet& sites){
    double tot = 0;

    for(auto i : range1(length(psi)) ){
        //position call very important! otherwise one would need to 
        //contract the whole tensor network of <psi|O|psi>
        //this way, only the local operator at site i is needed
        psi.position(i);

        auto val = psi.A(i) * sites.op("Ntot",i)* dag(prime(psi.A(i),"Site"));
        std::cout << std::real(val.cplx()) << " ";
        tot += std::real(val.cplx());
    }
    std::cout << std::endl;
    Print(tot);
}

//fills the vectors eps and V with the correct values for given gamma and number of bath sites
void GetBathParams(double U, double gamma, std::vector<double>& eps, std::vector<double>& V, int NBath){
    double dEnergy = 2./NBath;
    double Vval = std::sqrt( 2*gamma/NBath );

    V.resize(0);
    eps.resize(0);

    eps.push_back(-U/2.);
    V.push_back(0.);

    for(auto k: range1(NBath)){
        eps.push_back( -1 + (k-0.5)*dEnergy );
        V.push_back( Vval );
    }

}

//alternative DMRG method no real need to use it I think
void MyDMRG(MPS & psi, MPO& H, double& energy, Args args){

    int max_sweeps = args.getInt("maxsweeps",20);
    double GS_CuO = args.getReal("Cutoff", 1E-9);
    energy =100;
    int N = length(H);
    LocalMPO Heff(H);
    //apply MPO once, to avoid non-convergence due to special form of the product state
    
    Print("Start DMRG");
    psi.position(1); Heff.position(1,psi);
    //args.add("Cutoff",1E-5);
    
    for(int sweep=1;sweep<=max_sweeps;sweep++){
        //if(sweep==5)  { args.add("Cutoff",1E-7);       }
        //if(sweep==25) { args.add("Cutoff",1E-10);      }
        //if(sweep==35) { args.add("Cutoff",GS_CuO);     }
        
        for(int k=1;k<=N-1;k++){

            psi.position(k); Heff.position(k,psi);
            auto phi = psi.A(k)*psi.A(k+1);

            //Solve effective eigenvalue problem
            if(sweep<25) { energy = davidson(Heff,phi); }
            else { energy = forktps::lanzcos(Heff,phi); }
            
            //svd back
            Spectrum spec=psi.svdBond(k,phi,Fromleft,Heff,args);
            
            psi.normalize();
        }
        
        for(int k=N-2;k>=1;k--){
            psi.position(k); Heff.position(k,psi);
            
            auto phi = psi.A(k)*psi.A(k+1);
            //Solve effective eigenvalue problem
            if(sweep<25) { energy = davidson(Heff,phi); }
            else { energy = forktps::lanzcos(Heff,phi); }

            //svd back
            Spectrum spec=psi.svdBond(k,phi,Fromleft,Heff,args);
            psi.normalize();
        }
        
        std::cout<<std::setprecision(20)<<"     sweep "<<sweep<< " the energy is: "<<energy<<std::endl;
    }

}