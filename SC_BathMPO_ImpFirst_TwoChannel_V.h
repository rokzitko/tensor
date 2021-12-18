inline void Fill_SCBath_MPO_ImpFirst_TwoChannel_V(MPO& H, const double Eshift, double epsishift1, double epsishift2, double epseff, const std::vector<double>& eps_,
                const std::vector<double>& v_, const params &p)
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
                                qn0,       2, Out, "Link" ));

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

        W += p.sites.op("Ntot",i)  * setElt(right(2)) * epseff; // due to V the impurity level energy also shifts!
        W += p.sites.op("Nup",i)  * setElt(right(2)) * p.qd->EZ()/2.0; // impurity Zeeman energy
        W += p.sites.op("Ndn",i)  * setElt(right(2)) * (-1) * p.qd->EZ()/2.0; // impurity Zeeman energy
        W += p.sites.op("Nupdn",i) * setElt(right(2)) * p.qd->U(); // not Ueff!
        W += p.sites.op("Id",i)    * setElt(right(2)) * Eshift;

        W += p.sites.op("Cup*F",i) * setElt(right(3))    * (-1);
        W += p.sites.op("Cdn*F",i) * setElt(right(4))    * (-1);
        W += p.sites.op("Cdagup*F",i) * setElt(right(5)) * (+1);
        W += p.sites.op("Cdagdn*F",i) * setElt(right(6)) * (+1);

        W += p.sites.op("Ntot",i) * setElt(right(9)) * p.V1imp;
        W += p.sites.op("Ntot",i) * setElt(right(10)) * p.V2imp;
    }

    // sites 2 ... N-1 are matrices
    for(auto i : range1(2,((length(H)+1)/2)-1 )){

        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps_[i-1] + epsishift1 + p.sc1->Ec()*(1.0-2.0*p.sc1->n0())); // !
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * p.sc1->EZ()/2.0; // bulk Zeeman energy
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1.) * p.sc1->EZ()/2.0; // bulk Zeeman energy
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (p.sc1->g() * pow(p.sc1->y(i-1), 2) + 2.0*p.sc1->Ec()); // !

        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(7)) * p.sc1->g() * p.sc1->y(i-1);
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(8)) * p.sc1->g() * p.sc1->y(i-1);
        W += p.sites.op("Ntot", i)          * setElt(left(1),right(9)) * 2.0*p.sc1->Ec(); // !

        W += p.sites.op("Id",i)*setElt(left(2),right(2));
        W += p.sites.op("F" ,i)*setElt(left(3),right(3));
        W += p.sites.op("F" ,i)*setElt(left(4),right(4));
        W += p.sites.op("F" ,i)*setElt(left(5),right(5));
        W += p.sites.op("F" ,i)*setElt(left(6),right(6));
        W += p.sites.op("Id",i)*setElt(left(7),right(7));
        W += p.sites.op("Id",i)*setElt(left(8),right(8));
        W += p.sites.op("Id",i)*setElt(left(9),right(9));
        W += p.sites.op("Id",i)*setElt(left(10),right(10));

        W += p.sites.op("Cdagup",i)*setElt(left(3),right(2))* v_[i-1];
        W += p.sites.op("Cdagdn",i)*setElt(left(4),right(2))* v_[i-1];
        W += p.sites.op("Cup",   i)*setElt(left(5),right(2))* v_[i-1];
        W += p.sites.op("Cdn",   i)*setElt(left(6),right(2))* v_[i-1];

        W += p.sites.op("Cdagup*Cdagdn",i)*setElt(left(7),right(2)) * p.sc1->y(i-1);
        W += p.sites.op("Cdn*Cup",i)      *setElt(left(8),right(2)) * p.sc1->y(i-1);
        W += p.sites.op("Ntot",i)         *setElt(left(9),right(2)); // !
    }

    // central matrix is the same as the ones for SC1, just connecting terms are zero to break up the SCs
    {
    	int i = ((length(H)+1)/2);

        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps_[i-1] + epsishift1 + p.sc1->Ec()*(1.0-2.0*p.sc1->n0())); // !
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * p.sc1->EZ()/2.0; // bulk Zeeman energy
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1.) * p.sc1->EZ()/2.0; // bulk Zeeman energy
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (p.sc1->g() * pow(p.sc1->y(i-1), 2) + 2.0*p.sc1->Ec()); // !

        W += p.sites.op("Id",i)*setElt(left(2),right(2));
        W += p.sites.op("F" ,i)*setElt(left(3),right(3));
        W += p.sites.op("F" ,i)*setElt(left(4),right(4));
        W += p.sites.op("F" ,i)*setElt(left(5),right(5));
        W += p.sites.op("F" ,i)*setElt(left(6),right(6));
        //W += p.sites.op("Id",i)*setElt(left(7),right(7));
        //W += p.sites.op("Id",i)*setElt(left(8),right(8));
        //W += p.sites.op("Id",i)*setElt(left(9),right(9));
        W += p.sites.op("Id",i)*setElt(left(10),right(10));

        W += p.sites.op("Cdagup",i)*setElt(left(3),right(2))* v_[i-1];
        W += p.sites.op("Cdagdn",i)*setElt(left(4),right(2))* v_[i-1];
        W += p.sites.op("Cup",   i)*setElt(left(5),right(2))* v_[i-1];
        W += p.sites.op("Cdn",   i)*setElt(left(6),right(2))* v_[i-1];

        W += p.sites.op("Cdagup*Cdagdn",i)*setElt(left(7),right(2)) * p.sc1->y(i-1);
        W += p.sites.op("Cdn*Cup",i)      *setElt(left(8),right(2)) * p.sc1->y(i-1);       
        W += p.sites.op("Ntot",i)         *setElt(left(9),right(2)); // !
    }

    // onwards the same, just for sc2
    int shift = (length(H)-1)/2; // number of matrices for one channel. i-shift is the index of the i-th level in the second channel.
    for(auto i : range1( ((length(H)+1)/2)+1, length(H)-1 )){

        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps_[i-1] + epsishift2 + p.sc2->Ec()*(1.0-2.0*p.sc2->n0())); // !
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * p.sc2->EZ()/2.0; // bulk Zeeman energy
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1.) * p.sc2->EZ()/2.0; // bulk Zeeman energy
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (p.sc2->g() * (p.sc2->y(i-1-shift), 2) + 2.0*p.sc2->Ec()); // !

        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(7)) * p.sc2->g() * p.sc2->y(i-1-shift);
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(8)) * p.sc2->g() * p.sc2->y(i-1-shift);
        W += p.sites.op("Ntot", i)          * setElt(left(1),right(10)) * 2.0*p.sc2->Ec(); // 10 here also!

        W += p.sites.op("Id",i)*setElt(left(2),right(2));
        W += p.sites.op("F" ,i)*setElt(left(3),right(3));
        W += p.sites.op("F" ,i)*setElt(left(4),right(4));
        W += p.sites.op("F" ,i)*setElt(left(5),right(5));
        W += p.sites.op("F" ,i)*setElt(left(6),right(6));
        W += p.sites.op("Id",i)*setElt(left(7),right(7));
        W += p.sites.op("Id",i)*setElt(left(8),right(8));
        
        W += p.sites.op("Id",i)*setElt(left(10),right(10)); // 10 here, to accumulate the sum of nSC2

        W += p.sites.op("Cdagup",i)*setElt(left(3),right(2))* v_[i-1];
        W += p.sites.op("Cdagdn",i)*setElt(left(4),right(2))* v_[i-1];
        W += p.sites.op("Cup",   i)*setElt(left(5),right(2))* v_[i-1];
        W += p.sites.op("Cdn",   i)*setElt(left(6),right(2))* v_[i-1];

        W += p.sites.op("Cdagup*Cdagdn",i)*setElt(left(7),right(2)) * p.sc2->y(i-1-shift);
        W += p.sites.op("Cdn*Cup",i)      *setElt(left(8),right(2)) * p.sc2->y(i-1-shift);

        W += p.sites.op("Ntot",i)         *setElt(left(10),right(2)); // this is dimension 10, as the sum_i n_i has to be transferred all the way to the imp site, in order to multiply it by nimp!
    
    }


    //site N is a vector again
    {
        int i = length(H);
        
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );

        W = ITensor(left, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Ntot",  i) * setElt(left(1)) * (eps_[i-1] + epsishift2 + p.sc2->Ec()*(1.0-2.0*p.sc2->n0())); // !
        W += p.sites.op("Nup",  i)  * setElt(left(1)) * p.sc2->EZ()/2.0; // bulk Zeeman energy
        W += p.sites.op("Ndn",  i)  * setElt(left(1)) * (-1) * p.sc2->EZ()/2.0; // bulk Zeeman energy
        W += p.sites.op("Nupdn",i)  * setElt(left(1)) * (p.sc2->g() * pow(p.sc2->y(i-1-shift), 2) + 2.0*p.sc2->Ec()); // !

        W += p.sites.op("Id",    i) * setElt(left(2)) ;
        W += p.sites.op("Cdagup",i) * setElt(left(3)) * v_[i-1];
        W += p.sites.op("Cdagdn",i) * setElt(left(4)) * v_[i-1];
        W += p.sites.op("Cup",   i) * setElt(left(5)) * v_[i-1];
        W += p.sites.op("Cdn",   i) * setElt(left(6)) * v_[i-1];

        W += p.sites.op("Cdagup*Cdagdn",i) * setElt(left(7)) * p.sc2->y(i-1-shift);
        W += p.sites.op("Cdn*Cup",      i) * setElt(left(8)) * p.sc2->y(i-1-shift);

        W += p.sites.op("Ntot",         i) * setElt(left(10)); // this is dimension 10, as it has to be transferred all the way to the imp site, in order to multiply it by nimp!
    }

  }
