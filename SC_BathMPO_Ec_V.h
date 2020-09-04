#include <vector>
using namespace itensor;

//fills the MPO tensors
void Fill_SCBath_MPO_Ec_V(MPO& H, const std::vector<double>& eps_,
                const std::vector<double>& v_, double epseff, double epsishift, const params &p)
{
      //QN objects are necessary to have abelian symmetries in MPS
      QN    qn0  ( {"Sz",  0},{"Nf", 0} ),
            cupC ( {"Sz", +1},{"Nf",+1} ),
            cdnC ( {"Sz", -1},{"Nf",+1} ),
            cupA ( {"Sz", -1},{"Nf",-1} ),
            cdnA ( {"Sz", +1},{"Nf",-1} );


    std::vector<Index> links;
    links.push_back( Index() );

    //first we create the link indices which carry quantum number information
    for(auto i : range1(length(H))){
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

        W += p.sites.op("Ntot",i)  * setElt(right(2)) * epseff; // not eps_[i-1], neither p.epsimp!!
        W += p.sites.op("Nup",i)  * setElt(right(2)) * p.qd->EZ(); // impurity Zeeman energy
        W += p.sites.op("Ndn",i)  * setElt(right(2)) * (-1) * p.qd->EZ(); // impurity Zeeman energy
        W += p.sites.op("Nupdn",i) * setElt(right(2)) * p.qd->U(); // not Ueff!

        W += p.sites.op("Cup*F",i) * setElt(right(3))    * (-1);
        W += p.sites.op("Cdn*F",i) * setElt(right(4))    * (-1);
        W += p.sites.op("Cdagup*F",i) * setElt(right(5)) * (+1);
        W += p.sites.op("Cdagdn*F",i) * setElt(right(6)) * (+1);

        if (p.V12 != 0.0)
          W += p.sites.op("Ntot",i) * setElt(right(9)) * p.V12;
    }

    // sites 2 ... N-1 are matrices
    for(auto i : range1(2,length(H)-1)){
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps_[i-1] + epsishift + p.Ec*(1.0-2.0*p.n0)); // !!
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * p.EZ_bulk; // bulk Zeeman energy
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1.) * p.EZ_bulk; // bulk Zeeman energy
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (p.g + 2.0*p.Ec); // !

        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(7)) * p.g;
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(8)) * p.g;
        W += p.sites.op("Ntot", i)          * setElt(left(1),right(9)) * 2.0*p.Ec; // !

        W += p.sites.op("Id",i)*setElt(left(2),right(2));
        W += p.sites.op("F" ,i)*setElt(left(3),right(3));
        W += p.sites.op("F" ,i)*setElt(left(4),right(4));
        W += p.sites.op("F" ,i)*setElt(left(5),right(5));
        W += p.sites.op("F" ,i)*setElt(left(6),right(6));
        W += p.sites.op("Id",i)*setElt(left(7),right(7));
        W += p.sites.op("Id",i)*setElt(left(8),right(8));
        W += p.sites.op("Id",i)*setElt(left(9),right(9));

        W += p.sites.op("Cdagup",i)*setElt(left(3),right(2))* v_[i-1];
        W += p.sites.op("Cdagdn",i)*setElt(left(4),right(2))* v_[i-1];
        W += p.sites.op("Cup",   i)*setElt(left(5),right(2))* v_[i-1];
        W += p.sites.op("Cdn",   i)*setElt(left(6),right(2))* v_[i-1];

        W += p.sites.op("Cdagup*Cdagdn",i)*setElt(left(7),right(2));
        W += p.sites.op("Cdn*Cup",i)      *setElt(left(8),right(2));
        W += p.sites.op("Ntot",i)         *setElt(left(9),right(2)); // !

        if (p.verbose) std::cout << "using " << eps_[i-1]+epsishift << " and "<<v_[i-1]<<std::endl;
    }

    //site N is a vector again
    {
        int i = length(H);
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );

        W = ITensor(left, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Ntot",  i) * setElt(left(1)) * (eps_[i-1] + epsishift + p.Ec*(1.0-2.0*p.n0)); // !!
        W += p.sites.op("Nup",  i)  * setElt(left(1)) * p.EZ_bulk; // bulk Zeeman energy
        W += p.sites.op("Ndn",  i)  * setElt(left(1)) * (-1) * p.EZ_bulk; // bulk Zeeman energy
        W += p.sites.op("Nupdn",i)  * setElt(left(1)) * (p.g + 2.0*p.Ec); // !

        W += p.sites.op("Id",    i) * setElt(left(2)) ;
        W += p.sites.op("Cdagup",i) * setElt(left(3)) * v_[i-1];
        W += p.sites.op("Cdagdn",i) * setElt(left(4)) * v_[i-1];
        W += p.sites.op("Cup",   i) * setElt(left(5)) * v_[i-1];
        W += p.sites.op("Cdn",   i) * setElt(left(6)) * v_[i-1];

        W += p.sites.op("Cdagup*Cdagdn",i) * setElt(left(7));
        W += p.sites.op("Cdn*Cup",      i) * setElt(left(8));
        W += p.sites.op("Ntot",         i) * setElt(left(9)); // !

        if (p.verbose) std::cout << "using " << eps_[i-1]+epsishift << " and "<<v_[i-1]<<std::endl;
    }

  }
