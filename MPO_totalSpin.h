inline void makeS2_MPO(MPO& H, params &p){

    // Inspired by http://itensor.org/support/641/to-measure-total-spin-angular-momentum-s-2-on-mps
    // Represents a MPO of the total spin operator S^2
    // Total spin is a vector sum of all local spin operators; S^2 = \sum_{ij} S_i^z S_j^z + 0.5 ( S_i^+ S_j^- + S_i^- S_j^+ )
    // iTensor works with spin in units of 1/2 (ie. the eigenvalue of S_z is 1), so the Sz operators are multiplied by 2.

    QN  ind0 ( {"Sz",  0},{"Nf", 0} ),
        indm ( {"Sz", -2},{"Nf", 0} ),
        indp ( {"Sz", +2},{"Nf", 0} );

    QN  ind_so ( {"Nf", 0} );
        

    //THIS DOES NOT WORK FOR SOME REASON :(
    if (!p.problem->spin_conservation()) {
        QN  ind0 ( {"Nf", 0} ),
            indm ( {"Nf", 0} ),
            indp ( {"Nf", 0} );
    }

    std::vector<Index> links;
    links.push_back( Index() );


    if (p.problem->spin_conservation()){
    
        for (auto i : range1(length(H)+1)){
            links.push_back(Index(  ind0,   3,
                                    indm,   1,
                                    indp,   1,
                                    Out, "Link" ));
        }
    }
    else {
        for (auto i : range1(length(H)+1)){
        links.push_back(Index(  ind_so,   5,
                                Out, "Link" ));
        }
    }

    {
        int i = 1;
        ITensor& W = H.ref(i);
        Index right = links.at(i);
    
        W = ITensor(right, p.sites.si(i), p.sites.siP(i) );        


        W += p.sites.op("Id", i) * setElt(right(1));
        W += p.sites.op("S2", i) * setElt(right(2));
        W += p.sites.op("Sz", i) * setElt(right(3)) * 2.;
        W += p.sites.op("S+", i) * setElt(right(4)) ;
        W += p.sites.op("S-", i) * setElt(right(5)) ;

    }

    for (auto i : range1(2, length(H)-1)){

        ITensor& W = H.ref(i);

        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, p.sites.si(i), p.sites.siP(i));
        
        W += p.sites.op("S2",i) * setElt(left(1), right(2)) ;
        W += p.sites.op("Sz",i) * setElt(left(1), right(3)) * 2.;
        W += p.sites.op("S+",i) * setElt(left(1), right(4)) ;
        W += p.sites.op("S-",i) * setElt(left(1), right(5)) ;

        W += p.sites.op("Sz",i) * setElt(left(3), right(2));
        W += p.sites.op("S-",i) * setElt(left(4), right(2));
        W += p.sites.op("S+",i) * setElt(left(5), right(2));

        W += p.sites.op("Id",i) * setElt(left(1), right(1));
        W += p.sites.op("Id",i) * setElt(left(2), right(2));
        W += p.sites.op("Id",i) * setElt(left(3), right(3));
        W += p.sites.op("Id",i) * setElt(left(4), right(4));
        W += p.sites.op("Id",i) * setElt(left(5), right(5));
    
    }
    
    {
        int i = length(H);
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        
        W = ITensor(left, p.sites.si(i), p.sites.siP(i) );

        W += p.sites.op("S2", i) * setElt(left(1));
        W += p.sites.op("Id", i) * setElt(left(2));
        W += p.sites.op("Sz", i) * setElt(left(3));
        W += p.sites.op("S-", i) * setElt(left(4));
        W += p.sites.op("S+", i) * setElt(left(5));
    
    }


}