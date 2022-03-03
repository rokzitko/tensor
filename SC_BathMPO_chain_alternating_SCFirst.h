inline void impurityTensor(MPO &H, const int i, const int imp_num, const std::vector<Index> links, const params &p){
    // i is the index of the tensor in the MPO, imp_num numbers the impurity

    const auto &IMP = p.chain_imps[imp_num-1]; // 0-based vector!

    std::cout << "this is imp on site " << i << "\n";
    std::cout << "parameters: eps: " << IMP->eps() << " U: " << IMP->U() << "\n";

    ITensor& W = H.ref(i);
    Index left = dag( links.at(i-1) );
    Index right = links.at(i);

    W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

    W += p.sites.op("Id",i) * setElt(left(1), right(1));
    W += p.sites.op("Id",i) * setElt(left(2), right(2));

    W += p.sites.op("Ntot",i)  * setElt(left(1), right(2)) * IMP->eps();
    W += p.sites.op("Nup",i)   * setElt(left(1), right(2)) * IMP->EZ()/2.0;
    W += p.sites.op("Ndn",i)   * setElt(left(1), right(2)) * (-1) * IMP->EZ()/2.0;
    W += p.sites.op("Nupdn",i) * setElt(left(1), right(2)) * IMP->U();
    
    // hybridizations to the right
    W += p.sites.op("Cup*F",    i)*setElt(left(1),right(7)) * (-1);
    W += p.sites.op("Cdn*F",    i)*setElt(left(1),right(8)) * (-1);
    W += p.sites.op("Cdagup*F", i)*setElt(left(1),right(9)) * (+1);
    W += p.sites.op("Cdagdn*F", i)*setElt(left(1),right(10))* (+1);

    // hybridizations to the left
    W += p.sites.op("Cup", i)     *setElt(left(3),right(2));
    W += p.sites.op("Cdn", i)     *setElt(left(4),right(2));
    W += p.sites.op("Cdagup", i)  *setElt(left(5),right(2));
    W += p.sites.op("Cdagdn", i)  *setElt(left(6),right(2));

}

inline void middleSC(MPO &H, const int first_site, const int SC_num, const std::vector<Index> links, const params &p){

    // these vectors are 0-based!!!
    const auto &SC = p.chain_scis[SC_num-1];
    const auto &eps = SC->eps(p.band_level_shift, p.flat_band, p.flat_band_factor, p.band_rescale);

    const auto &HYB_l = p.chain_hybs[2*(SC_num - 2) + 1]; 
    const auto &vl = HYB_l->V(SC->NBath());

    const auto &HYB_r = p.chain_hybs[2*(SC_num - 2) + 1 + 1];
    const auto &vr = HYB_r->V(SC->NBath());

    int j = 0;
    for (auto i : range1(first_site, first_site + p.SClevels - 1)){ // range1() IS INCLUSIVE, THUS -1!
        j ++; // index of the level in this SC island  

        std::cout << "vL on site " << i << " is " << vl[j] << ", j is " << j << "\n";
        std::cout << "vR on site " << i << " is " << vr[j] << ", j is " << j << "\n";
        std::cout << "eps on site " << i << " is " << eps[j] << "\n";

        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        // local H on site
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps[j] + SC->Ec()*(1-2*SC->n0())); // here use index i
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * SC->EZ()/2.0;
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1) * SC->EZ()/2.0;
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (SC->g() * pow(SC->y(j), 2) + 2*SC->Ec());

        // hybridizations to the right
        W += p.sites.op("Cdagup*F",i)*setElt(left(1),right(3))* (+vr[j]);
        W += p.sites.op("Cdagdn*F",i)*setElt(left(1),right(4))* (+vr[j]);
        W += p.sites.op("Cup*F",   i)*setElt(left(1),right(5))* (-vr[j]);
        W += p.sites.op("Cdn*F",   i)*setElt(left(1),right(6))* (-vr[j]);

        // hybridizations to the left
        W += p.sites.op("Cdagup",i)*setElt(left(7),right(2))* (+vl[j]);
        W += p.sites.op("Cdagdn",i)*setElt(left(8),right(2))* (+vl[j]);
        W += p.sites.op("Cup",   i)*setElt(left(9),right(2))* (+vl[j]);
        W += p.sites.op("Cdn",   i)*setElt(left(10),right(2))*(+vl[j]);

        //SC pairing
        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(11)) * SC->g() * SC->y(j);
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(12)) * SC->g() * SC->y(j);
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(13)) * 2*SC->Ec();

        // keep terms
        W += p.sites.op("Id",i)*setElt(left(2),right(2));

        W += p.sites.op("F",i)*setElt(left(7),right(7));
        W += p.sites.op("F",i)*setElt(left(8),right(8));
        W += p.sites.op("F",i)*setElt(left(9),right(9));
        W += p.sites.op("F",i)*setElt(left(10),right(10));

        W += p.sites.op("Id",i)*setElt(left(11),right(11));
        W += p.sites.op("Id",i)*setElt(left(12),right(12));
        W += p.sites.op("Id",i)*setElt(left(13),right(13));
    }
}



inline void Fill_SC_BathMPO_chain_alternating_SCFirst(MPO& H, const double Eshift, const params &p)
{

  // QN objects are necessary to have abelian symmetries in MPS
  // automatically find the correct values
  QN    qn0  = - div( p.sites.op( "Id",      1) ),
        cupC = - div( p.sites.op( "Cdagup",  1) ),
        cdnC = - div( p.sites.op( "Cdagdn",  1) ),
        cupA = - div( p.sites.op( "Cup",     1) ),
        cdnA = - div( p.sites.op( "Cdn",     1) );

  std::vector<Index> links;
  links.push_back( Index() );

    //first we create the link indices which carry quantum number information
    // All have the same link indeces, except the last QD-SC pair!
    for(auto i : range1(p.N - ( p.SClevels + 1))){
        links.push_back(Index(  qn0,       2,
                                cupC,      1,
                                cdnC,      1,
                                cupA,      1,
                                cdnA,      1,

                                cupA,      1,
                                cdnA,      1,
                                cupC,      1,
                                cdnC,      1,

                                cupA+cdnA, 1,
                                cupC+cdnC, 1,
                                qn0,       1, Out, "Link" ));

        }

    for(auto i : range1(p.SClevels + 1)){
        links.push_back(Index(  qn0,       2,
                                cupA,      1,
                                cdnA,      1,
                                cupC,      1,
                                cdnC,      1,

                                cupC,      1,
                                cdnC,      1,
                                cupA,      1,
                                cdnA,      1,

                                cupA+cdnA, 1,
                                cupC+cdnC, 1,
                                qn0,       1, Out, "Link" ));

        }

    // for the first SC in the chain
    const auto & SC1 = p.chain_scis[0];
    auto eps = SC1->eps(p.band_level_shift, p.flat_band, p.flat_band_factor, p.band_rescale);
    
    const auto & HYB_r = p.chain_hybs[0]; 
    auto vr = HYB_r->V(SC1->NBath());

    int j = 1; //this is the sc level counter, while i is the global i counter    
    //first site is a vector:
    {        
        int i = 1;
        ITensor& W = H.ref(i);
        Index right = links.at(i);

        W = ITensor(right, p.sites.si(i), p.sites.siP(i) );
        W += p.sites.op("Id",i) * setElt(right(1));

        std::cout << "v on site " << i << " is " << vr[j] << ", j is " << j << "\n";
        std::cout << "eps on site " << i << " is " << eps[j] << "\n";


        // local H on site
        W += p.sites.op("Ntot",i)  * setElt(right(2)) * (eps[j] + SC1->Ec()*(1-2*SC1->n0())); // here use index i
        W += p.sites.op("Nup",i)   * setElt(right(2)) * SC1->EZ()/2.0;
        W += p.sites.op("Ndn",i)   * setElt(right(2)) * (-1) * SC1->EZ()/2.0;
        W += p.sites.op("Nupdn",i) * setElt(right(2)) * (SC1->g() * pow(SC1->y(j), 2) + 2*SC1->Ec());
        W += p.sites.op("Id",i)    * setElt(right(2)) * Eshift;

        //hybridization to the right
        W += p.sites.op("Cdagup*F",i) * setElt(right(3))* (+vr[j]);
        W += p.sites.op("Cdagdn*F",i) * setElt(right(4))* (+vr[j]);
        W += p.sites.op("Cup*F",   i) * setElt(right(5))* (-vr[j]);
        W += p.sites.op("Cdn*F",   i) * setElt(right(6))* (-vr[j]);

        //SC pairing
        W += p.sites.op("Cdn*Cup",       i) * setElt(right(11)) * SC1->g() * SC1->y(j);
        W += p.sites.op("Cdagup*Cdagdn", i) * setElt(right(12)) * SC1->g() * SC1->y(j);
        W += p.sites.op("Ntot",i)           * setElt(right(13)) * 2*SC1->Ec();
    }

    // sites for the first SC
    for(auto i : range1(2, p.SClevels )){
        j++;

        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        std::cout << "v on site " << i << " is " << vr[j] << ", j is " << j << "\n";
        std::cout << "eps on site " << i << " is " << eps[j] << "\n";

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        // local H on site
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps[j] + SC1->Ec()*(1-2*SC1->n0())); // here use index i
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * SC1->EZ()/2.0;
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1) * SC1->EZ()/2.0;
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (SC1->g() * pow(SC1->y(j), 2) + 2*SC1->Ec());

        // hybridizations to the right
        W += p.sites.op("Cdagup*F",i)*setElt(left(1),right(3))* (+vr[j]);
        W += p.sites.op("Cdagdn*F",i)*setElt(left(1),right(4))* (+vr[j]);
        W += p.sites.op("Cup*F",   i)*setElt(left(1),right(5))* (-vr[j]);
        W += p.sites.op("Cdn*F",   i)*setElt(left(1),right(6))* (-vr[j]);

        //SC pairing
        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(11)) * SC1->g() * SC1->y(j);
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(12)) * SC1->g() * SC1->y(j);
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(13)) * 2*SC1->Ec();

        // keep terms
        W += p.sites.op("Id",i)*setElt(left(2),right(2));
        W += p.sites.op("F" ,i)*setElt(left(3),right(3));
        W += p.sites.op("F" ,i)*setElt(left(4),right(4));
        W += p.sites.op("F" ,i)*setElt(left(5),right(5));
        W += p.sites.op("F" ,i)*setElt(left(6),right(6));
        W += p.sites.op("Id",i)*setElt(left(10),right(10));
        W += p.sites.op("Id",i)*setElt(left(11),right(11));
        W += p.sites.op("Id",i)*setElt(left(12),right(12));

        // add SC pairing terms
        W += p.sites.op("Cdagup*Cdagdn",i)*setElt(left(11),right(2)) * SC1->y(j);
        W += p.sites.op("Cdn*Cup",i)      *setElt(left(12),right(2)) * SC1->y(j);
        W += p.sites.op("Ntot",i)         *setElt(left(13),right(2));
    }

    // Now add the QD-SC pairs. There are (chainLen - 3)/2 of them - the last QD-SC is a bit different and added separately
    std::cout << "THERE SHOULD BE NOTHING HERE FOR N=3\n";
    for (int pair = 1; pair <= (p.chainLen-3)/2; pair++){
        std::cout << ((p.SClevels + 1) * pair) << "\n";
        std::cout << ((p.SClevels + 1) * pair) + 1 << "\n";

        impurityTensor(H, ((p.SClevels + 1) * pair), pair, links, p);
        middleSC(H, ((p.SClevels + 1) * pair) + 1 , pair+1, links, p);
    }

    // add the last QD

    const auto &IMP = p.chain_imps[p.chain_imps.size()-1];

    int i = p.N - p.SClevels;

    std::cout << "this is last imp on site " << i << "\n";
    std::cout << "parameters: eps: " << IMP->eps() << " U: " << IMP->U() << "\n";


    ITensor& W = H.ref(i);
    Index left = dag( links.at(i-1) );
    Index right = links.at(i);

    W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

    W += p.sites.op("Id",i) * setElt(left(1), right(1));
    W += p.sites.op("Id",i) * setElt(left(2), right(2));

    W += p.sites.op("Ntot",i)  * setElt(left(1), right(2)) * IMP->eps();
    W += p.sites.op("Nup",i)   * setElt(left(1), right(2)) * IMP->EZ()/2.0;
    W += p.sites.op("Ndn",i)   * setElt(left(1), right(2)) * (-1) * IMP->EZ()/2.0;
    W += p.sites.op("Nupdn",i) * setElt(left(1), right(2)) * IMP->U();

    // hybridizations to the right
    W += p.sites.op("Cup*F",    i)*setElt(left(1),right(3)) * (-1);
    W += p.sites.op("Cdn*F",    i)*setElt(left(1),right(4)) * (-1);
    W += p.sites.op("Cdagup*F", i)*setElt(left(1),right(5)) * (+1);
    W += p.sites.op("Cdagdn*F", i)*setElt(left(1),right(6)) * (+1);

    // hybridizations to the left
    W += p.sites.op("Cup", i)     *setElt(left(3),right(2));
    W += p.sites.op("Cdn", i)     *setElt(left(4),right(2));
    W += p.sites.op("Cdagup", i)  *setElt(left(5),right(2));
    W += p.sites.op("Cdagdn", i)  *setElt(left(6),right(2));


    // add the last SC
    const auto & SCN = p.chain_scis[p.chain_scis.size()-1];
    eps = SCN->eps(p.band_level_shift, p.flat_band, p.flat_band_factor, p.band_rescale);

    const auto & HYB_l = p.chain_hybs[p.chain_hybs.size()-1];
    auto vl = HYB_l->V(SCN->NBath());

    j = 0; //this is the sc level counter, while i is the global i counter
    for (int i : range1( p.N - p.SClevels + 1, p.N-1 )){
        j++;

        std::cout << "v on site " << i << " is " << vl[j] << ", j is " << j << "\n";
        std::cout << "eps on site " << i << " is " << eps[j] << "\n";

        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Id",i) * setElt(left(1), right(1));

        // local H on site
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(2)) * (eps[j] + SCN->Ec()*(1-2*SCN->n0()));
        W += p.sites.op("Nup",i)            * setElt(left(1),right(2)) * SCN->EZ()/2.0;
        W += p.sites.op("Ndn",i)            * setElt(left(1),right(2)) * (-1) * SCN->EZ()/2.0;
        W += p.sites.op("Nupdn",i)          * setElt(left(1),right(2)) * (SCN->g() * pow(SCN->y(j), 2) + 2.0*SCN->Ec());

        // hybridizations to the left
        W += p.sites.op("Cdagup",i)*setElt(left(3),right(2))* vl[j];
        W += p.sites.op("Cdagdn",i)*setElt(left(4),right(2))* vl[j];
        W += p.sites.op("Cup",   i)*setElt(left(5),right(2))* vl[j];
        W += p.sites.op("Cdn",   i)*setElt(left(6),right(2))* vl[j];

        // SC pairing
        W += p.sites.op("Cdn*Cup",i)        * setElt(left(1),right(11)) * SCN->g() * SCN->y(j);
        W += p.sites.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(12)) * SCN->g() * SCN->y(j);
        W += p.sites.op("Ntot",i)           * setElt(left(1),right(13)) * 2.0*SCN->Ec(); // MISSING 2.0 FIXED !!

        W += p.sites.op("Id",i)*setElt(left(2),right(2));
        W += p.sites.op("F" ,i)*setElt(left(3),right(3));
        W += p.sites.op("F" ,i)*setElt(left(4),right(4));
        W += p.sites.op("F" ,i)*setElt(left(5),right(5));
        W += p.sites.op("F" ,i)*setElt(left(6),right(6));
        W += p.sites.op("Id",i)*setElt(left(11),right(11));
        W += p.sites.op("Id",i)*setElt(left(12),right(12));
        W += p.sites.op("Id",i)*setElt(left(13),right(13));

        W += p.sites.op("Cdagup*Cdagdn",i) * setElt(left(11),right(2)) * SCN->y(j);
        W += p.sites.op("Cdn*Cup",i)       * setElt(left(12),right(2)) * SCN->y(j);
        W += p.sites.op("Ntot",i)          * setElt(left(13),right(2));
    }

    //site N is a vector again - same as before
    {
        j++;
        int i = p.N;

        std::cout << "v on site " << i << " is " << vl[j] << ", j is " << j << "\n";


        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );

        W = ITensor(left, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("Ntot", i)  * setElt(left(1)) * (eps[j] + SCN->Ec()*(1-2*SCN->n0())); // use index i-1
        W += p.sites.op("Nup",  i)  * setElt(left(1)) * SCN->EZ()/2.0;
        W += p.sites.op("Ndn",  i)  * setElt(left(1)) * (-1) * SCN->EZ()/2.0;
        W += p.sites.op("Nupdn",i)  * setElt(left(1)) * (SCN->g() * pow(SCN->y(j), 2) + 2*SCN->Ec());

        W += p.sites.op("Id",    i) * setElt(left(2)) ;
        W += p.sites.op("Cdagup",i) * setElt(left(3)) * vl[j];
        W += p.sites.op("Cdagdn",i) * setElt(left(4)) * vl[j];
        W += p.sites.op("Cup",   i) * setElt(left(5)) * vl[j];
        W += p.sites.op("Cdn",   i) * setElt(left(6)) * vl[j];

        W += p.sites.op("Cdagup*Cdagdn",i) * setElt(left(11)) * SCN->y(j);
        W += p.sites.op("Cdn*Cup",      i) * setElt(left(12)) * SCN->y(j);
        W += p.sites.op("Ntot",         i) * setElt(left(13));
    }
  }

