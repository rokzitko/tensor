inline void Fill_SCBath_MPO_MiddleImp_TwoChannel(MPO& H, const double Eshift, const std::vector<double>& eps_,
                                                 const std::vector<double>& v_, const params &p)
{
  Expects(odd(length(H)));
  // QN objects are necessary to have abelian symmetries in MPS
  // automatically find the correct values
  QN    qn0  = - div( p.sites.op( "Id",      1) ),
    cupC = - div( p.sites.op( "Cdagup",  1) ),
    cdnC = - div( p.sites.op( "Cdagdn",  1) ),
    cupA = - div( p.sites.op( "Cup",     1) ),
    cdnA = - div( p.sites.op( "Cdn",     1) );

  //std::cout << "g1="  << p.sc1->g() << std::endl;
  std::cout << "Ec1=" << p.sc1->Ec() << std::endl;
  //std::cout << "g2="  << p.sc2->g() << std::endl;
  std::cout << "Ec2=" << p.sc2->Ec() << std::endl;

  std::vector<Index> links;
  links.push_back( Index() );

    //first we create the link indices which carry quantum number information
    for(auto i : range1( p.impindex-1 )){
        links.push_back(Index(  qn0,       2,
                                cupC,      1,
                                cdnC,      1,
                                cupA,      1,
                                cdnA,      1,
                                cupA+cdnA, 1,
                                cupC+cdnC, 1,
                                qn0,       1, Out, "Link" ));

    }
    //first we create the link indices which carry quantum number information
    for(auto i : range1( p.impindex, length(H)-1 )){
        links.push_back(Index(  qn0,       2,
                                cupA,      1,
                                cdnA,      1,
                                cupC,      1,
                                cdnC,      1,
                                cupA+cdnA, 1,
                                cupC+cdnC, 1,
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
        W += p.sites.op("Ntot",i)  * setElt(right(2)) * (eps_[i] + p.sc1->Ec()*(1-2*p.sc1->n0())); // here use index i
        W += p.sites.op("Nup",i)   * setElt(right(2)) * p.sc1->EZ()/2.0;
        W += p.sites.op("Ndn",i)   * setElt(right(2)) * (-1) * p.sc1->EZ()/2.0;
        W += p.sites.op("Nupdn",i) * setElt(right(2)) * (p.sc1->g() * pow(p.sc1->y(i), 2) + 2*p.sc1->Ec());

        //hybridization
        W += p.sites.op("Cdagup*F",i) * setElt(right(3))* (+v_[i]); // here use index i
        W += p.sites.op("Cdagdn*F",i) * setElt(right(4))* (+v_[i]); // here use index i
        W += p.sites.op("Cup*F",   i) * setElt(right(5))* (-v_[i]); // here use index i
        W += p.sites.op("Cdn*F",   i) * setElt(right(6))* (-v_[i]); // here use index i

        //SC pairing
        W += p.sites.op("Cdn*Cup",       i) * setElt(right(7)) * p.sc1->g() * p.sc1->y(i);
        W += p.sites.op("Cdagup*Cdagdn", i) * setElt(right(8)) * p.sc1->g() * p.sc1->y(i);

        W += p.sites.op("Ntot",i) * setElt(right(9)) * 2*p.sc1->Ec();
    }


    // sites 2 ... p.impindex-1 are matrices
    for(auto i : range1(2, p.impindex-1 )){
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        // local H on site
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps_[i] + p.sc1->Ec()*(1-2*p.sc1->n0())); // here use index i
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * p.sc1->EZ()/2.0;
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1) * p.sc1->EZ()/2.0;
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (p.sc1->g() * pow(p.sc1->y(i), 2) + 2*p.sc1->Ec());

        // hybridizations
        W += p.sites.op("Cdagup*F",i)*setElt(left(1),right(3))* (+v_[i] ); // here use index i
        W += p.sites.op("Cdagdn*F",i)*setElt(left(1),right(4))* (+v_[i] ); // here use index i
        W += p.sites.op("Cup*F",   i)*setElt(left(1),right(5))* (-v_[i] ); // here use index i
        W += p.sites.op("Cdn*F",   i)*setElt(left(1),right(6))* (-v_[i] ); // here use index i

        //SC pairing
        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(7)) * p.sc1->g() * p.sc1->y(i);
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(8)) * p.sc1->g() * p.sc1->y(i);
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(9)) * 2*p.sc1->Ec();

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
        W += p.sites.op("Cdagup*Cdagdn",i)*setElt(left(7),right(2)) * p.sc1->y(i);
        W += p.sites.op("Cdn*Cup",i)      *setElt(left(8),right(2)) * p.sc1->y(i);
        W += p.sites.op("Ntot",i)         *setElt(left(9),right(2)); // !!!
    }

    // impurity
    {
        int i = p.impindex;
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        W += p.sites.op("Ntot",i)  * setElt(left(1), right(2)) * p.qd->eps(); //CHECK THIS
        W += p.sites.op("Nup",i)   * setElt(left(1), right(2)) * p.qd->EZ()/2.0;
        W += p.sites.op("Ndn",i)   * setElt(left(1), right(2)) * (-1) * p.qd->EZ()/2.0;
        W += p.sites.op("Nupdn",i) * setElt(left(1), right(2)) * p.qd->U();
        W += p.sites.op("Id",i)    * setElt(left(1), right(2)) * Eshift;

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
    }

    // sites p.impindex+1 ... N -1 are the same as before
    int shift = (length(H)-1)/2; // number of matrices for one channel. i-shift is the index of the i-th level in the second channel.
    for(auto i : range1(p.impindex+1, length(H)-1 )){
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps_[i-1] + p.sc2->Ec()*(1-2*p.sc2->n0()));  // use index i-1
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * p.sc2->EZ()/2.0;
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1) * p.sc2->EZ()/2.0;
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (p.sc2->g() * pow(p.sc2->y(i-1-shift), 2) + 2.0*p.sc2->Ec()); // MISSING 2.0 FIXED !!

        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(7)) * p.sc2->g() * p.sc2->y(i-1-shift);
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(8)) * p.sc2->g() * p.sc2->y(i-1-shift);
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(9)) * 2.0*p.sc2->Ec(); // MISSING 2.0 FIXED !!

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

        W += p.sites.op("Cdagup*Cdagdn",i) * setElt(left(7),right(2)) * p.sc2->y(i-1-shift);
        W += p.sites.op("Cdn*Cup",i)       * setElt(left(8),right(2)) * p.sc2->y(i-1-shift);
        W += p.sites.op("Ntot",i)          * setElt(left(9),right(2));
    }

    //site N is a vector again - same as before
    {
        int i = length(H);
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );

        W = ITensor(left, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Ntot",  i) * setElt(left(1)) * (eps_[i-1] + p.sc2->Ec()*(1-2*p.sc2->n0())); // use index i-1
        W += p.sites.op("Nup",  i)  * setElt(left(1)) * p.sc2->EZ()/2.0;
        W += p.sites.op("Ndn",  i)  * setElt(left(1)) * (-1) * p.sc2->EZ()/2.0;
        W += p.sites.op("Nupdn",i)  * setElt(left(1)) * (p.sc2->g() * pow(p.sc2->y(i-1-shift), 2) + 2*p.sc2->Ec());

        W += p.sites.op("Id",    i) * setElt(left(2)) ;
        W += p.sites.op("Cdagup",i) * setElt(left(3)) * v_[i-1];  // use index i-1
        W += p.sites.op("Cdagdn",i) * setElt(left(4)) * v_[i-1];  // use index i-1
        W += p.sites.op("Cup",   i) * setElt(left(5)) * v_[i-1];  // use index i-1
        W += p.sites.op("Cdn",   i) * setElt(left(6)) * v_[i-1];  // use index i-1

        W += p.sites.op("Cdagup*Cdagdn",i) * setElt(left(7)) * p.sc2->y(i-1-shift);
        W += p.sites.op("Cdn*Cup",      i) * setElt(left(8)) * p.sc2->y(i-1-shift);
        W += p.sites.op("Ntot",         i) * setElt(left(9));
    }
  }

