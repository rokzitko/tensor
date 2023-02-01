inline void get_autoMPO_qd_sc_qd(MPO& H, const double Eshift, const params &p){

    auto ampo = AutoMPO(p.sites);

    const auto &vl = p.Gamma_L->V(p.sc->NBath());
    const auto &ivl = p.Gamma_L->iV(p.sc->NBath());
    const auto &vr = p.Gamma_R->V(p.sc->NBath());
    const auto &ivr = p.Gamma_R->iV(p.sc->NBath());

    ampo += Eshift,"Id",1;

    // left impurity, site 1
    ampo += p.qd_L->U(),"Nupdn",1;
    ampo += p.qd_L->eps(),"Ntot",1;

    // right impurity, site p.N
    ampo += p.qd_R->U(),"Nupdn",p.N;
    ampo += p.qd_R->eps(),"Ntot",p.N;

    // hopping terms     
    for(auto i: range1(2, p.N-1)){    
        ampo += (1_i * ivl[i-1] + vl[i-1]),"Cdagup",1,"Cup",i;
        ampo += (1_i * ivl[i-1] + vl[i-1]),"Cdagdn",1,"Cdn",i;
        ampo += (1_i * ivl[i-1] + vl[i-1]),"Cdagup",i,"Cup",1;
        ampo += (1_i * ivl[i-1] + vl[i-1]),"Cdagdn",i,"Cdn",1;
    }
    for(auto i: range1(2, p.N-1)){
        ampo += (1_i * ivr[i-1] + vr[i-1]),"Cdagup",p.N,"Cup",i;
        ampo += (1_i * ivr[i-1] + vr[i-1]),"Cdagdn",p.N,"Cdn",i;
        ampo += (1_i * ivr[i-1] + vr[i-1]),"Cdagup",i,"Cup",p.N;
        ampo += (1_i * ivr[i-1] + vr[i-1]),"Cdagdn",i,"Cdn",p.N;
    }

    // impurity hopping
    ampo += p.tQD,"Cdagup",p.N,"Cup",1;
    ampo += p.tQD,"Cdagdn",p.N,"Cdn",1;
    ampo += p.tQD,"Cdagup",1,"Cup",p.N;
    ampo += p.tQD,"Cdagdn",1,"Cdn",p.N;

    // bath terms
    int start = 2;
    int stop = p.N-1;
    const auto &eps = p.sc->eps(p.band_level_shift, p.flat_band, p.flat_band_factor, p.band_rescale);

    for(auto i: range1(start, stop)){
        ampo += eps[i-1],"Ntot",i;
        for(auto j: range1(start, stop)){
            ampo += p.sc->g() * p.sc->y(i-1) * p.sc->y(j-1),"Cdagup",i,"Cdagdn",i,"Cdn",j,"Cup",j;
        }
    }

    //occupation terms
    for(auto i: range1(start, stop)){
        ampo += - p.sc->Ec() * 2 * p.sc->n0(),"Ntot",i;
        
        for(auto j: range1(start, stop)){
            ampo += p.sc->Ec(),"Ntot",i,"Ntot",j;
        }
    } 
    H = toMPO(ampo);
}
