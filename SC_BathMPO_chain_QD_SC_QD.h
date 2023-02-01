inline void Fill_SCBath_MPO_qd_sc_qd(MPO& H, const double Eshift, const params &p){

    // parse eps and vs here to make it more readable
    const auto &eps = p.sc->eps(p.band_level_shift, p.flat_band, p.flat_band_factor, p.band_rescale);
    const auto &vl = p.Gamma_L->V(p.sc->NBath());
    const auto &ivl = p.Gamma_L->iV(p.sc->NBath());
    const auto &vr = p.Gamma_R->V(p.sc->NBath());
    const auto &ivr = p.Gamma_R->iV(p.sc->NBath());

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
    //therefore the tensor on the first and last site must be column/row vectors
    //and all sites between matrices
        
    //first site is the first QD, a vector:

    {
        int i = 1;
        
        ITensor& W = H.ref(i);
        Index right = links.at(i);

        W = ITensor(right, p.sites.si(i), p.sites.siP(i) );
        W += p.sites.op("Id",i) * setElt(right(1));

        W += p.sites.op("Ntot",i)   * setElt(right(2)) * p.qd_L->eps(); // not eps_[i-1]!
        W += p.sites.op("Nup",i)    * setElt(right(2)) * p.qd_L->EZ()/2.0; // impurity Zeeman energy
        W += p.sites.op("Ndn",i)    * setElt(right(2)) * (-1) * p.qd_L->EZ()/2.0; // impurity Zeeman energy
        W += p.sites.op("Nupdn",i)  * setElt(right(2)) * p.qd_L->U(); // not Ueff!
        W += p.sites.op("Id",i)     * setElt(right(2)) * Eshift;

        // these terms couple the two QDs
        W += p.sites.op("Cup",i)      * setElt(right(3)) * (-1) * p.tQD; // tQD is parsed separately! 
        W += p.sites.op("Cdn",i)      * setElt(right(4)) * (-1) * p.tQD;
        W += p.sites.op("Cdagup",i)   * setElt(right(5)) * (+1) * p.tQD;
        W += p.sites.op("Cdagdn",i)   * setElt(right(6)) * (+1) * p.tQD;

        // hopping between the left QD and the SC
        W += p.sites.op("Cup",i)      * setElt(right(7)) * (+1); 
        W += p.sites.op("Cdn",i)      * setElt(right(8)) * (+1);
        W += p.sites.op("Cdagup",i)   * setElt(right(9)) * (+1);
        W += p.sites.op("Cdagdn",i)   * setElt(right(10))* (+1);
    
    }

    // sites 2 ... N-1 are SC matrices
    for(auto i : range1(2,length(H)-1)){
        int j = i-1; // this is the index of the site in the SC
        
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);
        
        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        // local H on site
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps[j] + p.sc->Ec()*(1-2*p.sc->n0())); // here use index i-1, because i starts at 2
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * p.sc->EZ()/2.0;
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1) * p.sc->EZ()/2.0;
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (p.sc->g() * pow(p.sc->y(j), 2) + 2*p.sc->Ec());

        // hybridizations to the right
        W += p.sites.op("Cup*F",   i)*setElt(left(1),right(3))* (-vr[j] - 1_i * ivr[j]);
        W += p.sites.op("Cdn*F",   i)*setElt(left(1),right(4))* (-vr[j] - 1_i * ivr[j]);
        W += p.sites.op("Cdagup*F",i)*setElt(left(1),right(5))* (+vr[j] + 1_i * ivr[j]);
        W += p.sites.op("Cdagdn*F",i)*setElt(left(1),right(6))* (+vr[j] + 1_i * ivr[j]);

        // hybridizations to the left
        W += p.sites.op("Cdagup",i)*setElt(left(7),right(2))    * (+vl[j] + 1_i * ivl[j]);
        W += p.sites.op("Cdagdn",i)*setElt(left(8),right(2))    * (+vl[j] + 1_i * ivl[j]);
        W += p.sites.op("Cup",   i)*setElt(left(9),right(2))    * (+vl[j] + 1_i * ivl[j]);
        W += p.sites.op("Cdn",   i)*setElt(left(10),right(2))   * (+vl[j] + 1_i * ivl[j]);

        //SC pairing and Ec
        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(11)) * p.sc->g() * p.sc->y(j);
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(12)) * p.sc->g() * p.sc->y(j);
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(13)) * 2*p.sc->Ec();

        W += p.sites.op("Cdagup*Cdagdn",i) * setElt(left(11),right(2)) * p.sc->y(j);
        W += p.sites.op("Cdn*Cup",i)       * setElt(left(12),right(2)) * p.sc->y(j);
        W += p.sites.op("Ntot",i)          * setElt(left(13),right(2));

        // keep terms
        W += p.sites.op("Id",i)*setElt(left(2),right(2));

        W += p.sites.op("F",i)*setElt(left(3),right(3));
        W += p.sites.op("F",i)*setElt(left(4),right(4));
        W += p.sites.op("F",i)*setElt(left(5),right(5));
        W += p.sites.op("F",i)*setElt(left(6),right(6));
        W += p.sites.op("F",i)*setElt(left(7),right(7));
        W += p.sites.op("F",i)*setElt(left(8),right(8));
        W += p.sites.op("F",i)*setElt(left(9),right(9));
        W += p.sites.op("F",i)*setElt(left(10),right(10));
        
        W += p.sites.op("Id",i)*setElt(left(11),right(11));
        W += p.sites.op("Id",i)*setElt(left(12),right(12));
        W += p.sites.op("Id",i)*setElt(left(13),right(13));

    }

    //site N is the second QD, a vector again
    {
        int i = length(H);
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );

        W = ITensor(left, p.sites.si(i), p.sites.siP(i) );

        // first element is the local H
        W += p.sites.op("Ntot",i)   * setElt(left(1)) * p.qd_R->eps();
        W += p.sites.op("Nup",i)    * setElt(left(1)) * p.qd_R->EZ()/2.0; // impurity Zeeman energy
        W += p.sites.op("Ndn",i)    * setElt(left(1)) * (-1) * p.qd_R->EZ()/2.0; // impurity Zeeman energy
        W += p.sites.op("Nupdn",i)  * setElt(left(1)) * p.qd_R->U();

        W += p.sites.op("Id",i)  * setElt(left(2));

        // hopping to the left 
        W += p.sites.op("Cdagup", i)    * setElt(left(3));
        W += p.sites.op("Cdagdn", i)    * setElt(left(4));
        W += p.sites.op("Cup", i)       * setElt(left(5));
        W += p.sites.op("Cdn", i)       * setElt(left(6));
    }

  }