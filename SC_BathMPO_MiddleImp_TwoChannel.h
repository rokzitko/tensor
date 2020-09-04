#include <vector>
using namespace itensor;

//fills the MPO tensors
void Fill_SCBath_MPO_MiddleImp_TwoChannel(MPO& H, const std::vector<double>& eps_,
                const std::vector<double>& v_, const params &p)
{
      //QN objects are necessary to have abelian symmetries in MPS
      // automatically find the correct values
      QN    qn0  ( {"Sz",  0},{"Nf", 0} ),
            cupC ( {"Sz", +1},{"Nf",+1} ),
            cdnC ( {"Sz", -1},{"Nf",+1} ),
            cupA ( {"Sz", -1},{"Nf",-1} ),
            cdnA ( {"Sz", +1},{"Nf",-1} );

    std::vector<Index> links;
    links.push_back( Index() );

    if(length(H)%2 != 1){
        // currently only allow an odd total number of sites
        Error("Total number of sites should be odd so that the number of bath sites is even");
    }
    int impSite = std::round( (length(H)+1)/2 );

    //first we create the link indices which carry quantum number information
    for(auto i : range1( impSite-1 )){
        links.push_back(Index(  qn0,       2,
                                cupA,      1,
                                cdnA,      1,
                                cupC,      1,
                                cdnC,      1,
                                cupC+cdnC, 1,
                                cupA+cdnA, 1,
                                qn0,       1, Out, "Link" ));

    }
    //first we create the link indices which carry quantum number information
    for(auto i : range1( impSite, length(H)-1 )){
        links.push_back(Index(  qn0,       2,
                                cupC,      1,
                                cdnC,      1,
                                cupA,      1,
                                cdnA,      1,
                                cupC+cdnC, 1,
                                cupA+cdnA, 1,
                                qn0,       1, Out, "Link" ));

    }


    //then we just fill the MPO tensors which can be viewed
    //as matrices (vectors) of operators. if one multiplies
    //all matrices togehter one obtains the hamiltonian. 
    //therefore the tensor on the first and last site must be column/ row vectors
    //and all sites between matrices

    //first site is a vector:
    {
        int i = 1;
        ITensor& W = H.ref(i);
        Index right = links.at(i);


        W = ITensor(right, p.sites.si(i), p.sites.siP(i) );
        W += p.sites.op("Id",i) * setElt(right(1));


        // local H on site
        W += p.sites.op("Ntot",i)  * setElt(right(2)) * (eps_[i] + p.Ec1*(1-2*p.n01)); // here use index i
        W += p.sites.op("Nup",i)  * setElt(right(2)) * p.EZ_bulk1; 
        W += p.sites.op("Ndn",i)  * setElt(right(2)) * (-1) * p.EZ_bulk1;
        W += p.sites.op("Nupdn",i) * setElt(right(2)) * (p.g1 + 2*p.Ec1);

        //hybridization
        W += p.sites.op("Cdagup*F",i) * setElt(right(3))* (+v_[i]); // here use index i 
        W += p.sites.op("Cdagdn*F",i) * setElt(right(4))* (+v_[i]); // here use index i
        W += p.sites.op("Cup*F",   i) * setElt(right(5))* (-v_[i]); // here use index i
        W += p.sites.op("Cdn*F",   i) * setElt(right(6))* (-v_[i]); // here use index i

        //SC pairing
        W += p.sites.op("Cdn*Cup",      i) * setElt(right(7)) * p.g1;
        W += p.sites.op("Cdagup*Cdagdn",i) * setElt(right(8)) * p.g1;


        W += p.sites.op("Ntot",i) * setElt(right(9)) * 2*p.Ec1;

        if (p.verbose) std::cout << "using " << eps_[i] << " and "<<v_[i]<<std::endl;
    }


    // sites 2 ... impSite-1 are matrices
    for(auto i : range1(2, impSite-1 )){
        
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        // local H on site
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps_[i] + p.Ec1*(1-2*p.n01)); // here use index i
        W += p.sites.op("Nup",i)           * setElt(left(1),right(2)) * p.EZ_bulk1;
        W += p.sites.op("Ndn",i)           * setElt(left(1),right(2)) * (-1) * p.EZ_bulk1;
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (p.g1 + 2*p.Ec1);

        // hybridizations 
        W += p.sites.op("Cdagup*F",i)*setElt(left(1),right(3))* (+v_[i] ); // here use index i 
        W += p.sites.op("Cdagdn*F",i)*setElt(left(1),right(4))* (+v_[i] ); // here use index i
        W += p.sites.op("Cup*F",   i)*setElt(left(1),right(5))* (-v_[i] ); // here use index i
        W += p.sites.op("Cdn*F",   i)*setElt(left(1),right(6))* (-v_[i] ); // here use index i

        //SC pairing 
        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(7)) * p.g1;
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(8)) * p.g1;
        W += p.sites.op("Ntot",i)  * setElt(left(1),right(9)) * 2*p.Ec1;

        // keep terms
        W += p.sites.op("Id",i)*setElt(left(2),right(2));
        W += p.sites.op("F" ,i)*setElt(left(3),right(3));
        W += p.sites.op("F" ,i)*setElt(left(4),right(4));
        W += p.sites.op("F" ,i)*setElt(left(5),right(5));
        W += p.sites.op("F" ,i)*setElt(left(6),right(6));
        W += p.sites.op("Id",i)*setElt(left(7),right(7));
        W += p.sites.op("Id",i)*setElt(left(8),right(8));
        W += p.sites.op("Id",i)*setElt(left(9),right(9));

        // add SC pairing terms
        W += p.sites.op("Cdagup*Cdagdn",i)*setElt(left(7),right(2));
        W += p.sites.op("Cdn*Cup",i)      *setElt(left(8),right(2));
        W += p.sites.op("Ntot",i)      *setElt(left(9),right(2));

        if (p.verbose) std::cout << "using " << eps_[i] << " and "<<v_[i]<<std::endl;
    }

    // impurity 
    {
        int i = impSite;
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        W += p.sites.op("Ntot",i)  * setElt(left(1), right(2)) * p.epsimp; //CHECK THIS
        W += p.sites.op("Nup",i)  * setElt(left(1), right(2)) * p.EZ_imp;
        W += p.sites.op("Ndn",i)  * setElt(left(1), right(2)) * (-1) * p.EZ_imp;
        W += p.sites.op("Nupdn",i) * setElt(left(1), right(2)) * p.U;

        // hybridizations
        W += p.sites.op("Cup*F",    i)*setElt(left(1),right(3)) * (-1);
        W += p.sites.op("Cdn*F",    i)*setElt(left(1),right(4)) * (-1);
        W += p.sites.op("Cdagup*F", i)*setElt(left(1),right(5)) * (+1);
        W += p.sites.op("Cdagdn*F", i)*setElt(left(1),right(6)) * (+1);

        W += p.sites.op("Id",i) * setElt(left(2), right(2));


        // hybridizations
        W += p.sites.op("Cup",    i)*setElt(left(3),right(2));
        W += p.sites.op("Cdn",    i)*setElt(left(4),right(2));
        W += p.sites.op("Cdagup", i)*setElt(left(5),right(2));
        W += p.sites.op("Cdagdn", i)*setElt(left(6),right(2));
        //keep SC pairing
        W += p.sites.op("Id",i) *setElt(left(7),right(7)) * p.SCSCinteraction; // THESE THREE TERMS COUPLE THE SC1 AND SC2 LEVELS

        W += p.sites.op("Id",i) *setElt(left(8),right(8)) * p.SCSCinteraction;
        W += p.sites.op("Id",i) *setElt(left(9),right(9)) * p.SCSCinteraction; 

        if (p.verbose) std::cout << "using " << eps_[0] <<std::endl;
    }

    // sites impSite+1 ... N -1 are the same as before
    for(auto i : range1(impSite+1, length(H)-1 )){
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps_[i-1] + p.Ec2*(1-2*p.n02));  // use index i-1
        W += p.sites.op("Nup",i)           * setElt(left(1),right(2)) * p.EZ_bulk2;
        W += p.sites.op("Ndn",i)           * setElt(left(1),right(2)) * (-1) * p.EZ_bulk2;
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (p.g2 + p.Ec2);

        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(7)) * p.g2;
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(8)) * p.g2;
        W += p.sites.op("Ntot",i)  * setElt(left(1),right(9)) * p.Ec2;

        W += p.sites.op("Id",i)*setElt(left(2),right(2));
        W += p.sites.op("F" ,i)*setElt(left(3),right(3));
        W += p.sites.op("F" ,i)*setElt(left(4),right(4));
        W += p.sites.op("F" ,i)*setElt(left(5),right(5));
        W += p.sites.op("F" ,i)*setElt(left(6),right(6));
        W += p.sites.op("Id",i)*setElt(left(7),right(7));
        W += p.sites.op("Id",i)*setElt(left(8),right(8));
        W += p.sites.op("Id",i)*setElt(left(9),right(9));

        W += p.sites.op("Cdagup",i)*setElt(left(3),right(2))* v_[i-1];  // use index i-1
        W += p.sites.op("Cdagdn",i)*setElt(left(4),right(2))* v_[i-1];  // use index i-1
        W += p.sites.op("Cup",   i)*setElt(left(5),right(2))* v_[i-1];  // use index i-1
        W += p.sites.op("Cdn",   i)*setElt(left(6),right(2))* v_[i-1];  // use index i-1

        W += p.sites.op("Cdagup*Cdagdn",i)*setElt(left(7),right(2));
        W += p.sites.op("Cdn*Cup",i)      *setElt(left(8),right(2));
        W += p.sites.op("Ntot",i)      *setElt(left(9),right(2));

        if (p.verbose) std::cout << "using " << eps_[i-1] << " and "<<v_[i-1]<<std::endl;
    }

    //site N is a vector again - same as before
    {
        int i = length(H);
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );

        W = ITensor(left, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Ntot",  i) * setElt(left(1)) * (eps_[i-1] + p.Ec2*(1-2*p.n02)); // use index i-1
        W += p.sites.op("Nup",  i) * setElt(left(1)) * p.EZ_bulk2;
        W += p.sites.op("Ndn",  i) * setElt(left(1)) * (-1) * p.EZ_bulk2;
        W += p.sites.op("Nupdn",i)  * setElt(left(1)) * (p.g2 + 2*p.Ec2);

        W += p.sites.op("Id",    i) * setElt(left(2)) ;
        W += p.sites.op("Cdagup",i) * setElt(left(3)) * v_[i-1];  // use index i-1
        W += p.sites.op("Cdagdn",i) * setElt(left(4)) * v_[i-1];  // use index i-1
        W += p.sites.op("Cup",   i) * setElt(left(5)) * v_[i-1];  // use index i-1
        W += p.sites.op("Cdn",   i) * setElt(left(6)) * v_[i-1];  // use index i-1

        W += p.sites.op("Cdagup*Cdagdn",i) * setElt(left(7));
        W += p.sites.op("Cdn*Cup",      i) * setElt(left(8));
        W += p.sites.op("Ntot",      i) * setElt(left(9));

        if (p.verbose) std::cout << "using " << eps_[i-1] << " and "<<v_[i-1]<<std::endl;
    }
 

  }

