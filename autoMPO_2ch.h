
void get_MPO_bath(auto& ampo, const std::vector<double>& eps_, const std::vector<double>& v_, int startind, int endind, const auto& ch, const double shiftFactor){
   
    // bath terms     
    for(auto i: range1(startind, endind)){
        ampo += eps_[i-1],"Ntot",i;
        for(auto j: range1(startind, endind)){
            ampo += ch->g(i - shiftFactor -1),"Cdagup",i,"Cdagdn",i,"Cdn",j,"Cup",j;
        }
    }
    //hopping terms
    for(auto i: range1(startind, endind-1)){
        ampo += ch->t(),"Cdagup",i,"Cup",i+1;
        ampo += ch->t(),"Cdagdn",i,"Cdn",i+1;
        ampo += ch->t(),"Cdagup",i+1,"Cup",i;
        ampo += ch->t(),"Cdagdn",i+1,"Cdn",i;
    }

    //occupation terms
    for(auto i: range1(startind, endind)){
        ampo += - ch->Ec() * 2 * ch->n0(),"Ntot",i;
        
        for(auto j: range1(startind, endind)){
            ampo += ch->Ec(),"Ntot",i,"Ntot",j;
        }
    }        
    
    ampo += ch->Ec() * pow(ch->n0(), 2),"Id",1;

}

void capcacitive_coupling(auto& ampo, const double V, int startind, int endind, const auto& ch, const params &p){

    for(auto i: range1(startind, endind)){
        ampo += V,"Ntot",1,"Ntot",i;
        ampo += - V * p.qd->nu(),"Ntot",i;
    }
    ampo += - V * ch->n0(),"Ntot",1;
    ampo += V * p.qd->nu() * ch->n0(),"Id",1;

}

inline void get_autoMPO_2ch(MPO& H, const double Eshift, const std::vector<double>& eps_, const std::vector<double>& v_, const params &p){

    auto ampo = AutoMPO(p.sites);

    ampo += Eshift,"Id",1;

    // impurity term
    ampo += p.qd->U(),"Nupdn",1;
    ampo += p.qd->eps(),"Ntot",1;

    // impurity coupling terms     
    for(auto i: range1(2, p.N)){    
        ampo += v_[i-1],"Cdagup",1,"Cup",i;
        ampo += v_[i-1],"Cdagdn",1,"Cdn",i;
        ampo += v_[i-1],"Cdagup",i,"Cup",1;
        ampo += v_[i-1],"Cdagdn",i,"Cdn",1;
    }

    int sc1start = 2;
    int sc1end = ((p.N-1)/2) +1;
    int sc2start = ((p.N-1)/2) + 2;
    int sc2end = p.N;


    // capacative coupling
    capcacitive_coupling(ampo, p.V1imp, sc1start, sc1end, p.sc1, p);
    capcacitive_coupling(ampo, p.V2imp, sc2start, sc2end, p.sc2, p);

    get_MPO_bath(ampo, eps_, v_, sc1start, sc1end, p.sc1, 0);
    get_MPO_bath(ampo, eps_, v_, sc2start, sc2end, p.sc2, sc2start-1);
    

    H = toMPO(ampo);

}
