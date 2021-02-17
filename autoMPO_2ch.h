
void get_MPO_bath(auto& ampo, const std::vector<double>& eps_, const std::vector<double>& v_, int startind, int endind, const auto& ch){
   
    // bath terms     
    for(auto i: range1(startind, endind)){
        ampo += eps_[i-1],"Ntot",i;
        for(auto j: range1(startind, endind)){
            ampo += ch->g(),"Cdagup",i,"Cdagdn",i,"Cdn",j,"Cup",j;
        }
    }
    //hopping terms
    for(auto i: range1(startind, endind-1)){
        ampo += ch->t(),"Cdagup",i,"Cup",i+1;
        ampo += ch->t(),"Cdagdn",i,"Cdn",i+1;
        ampo += ch->t(),"Cdagup",i+1,"Cup",i;
        ampo += ch->t(),"Cdagdn",i+1,"Cdn",i;
    }    
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

    get_MPO_bath(ampo, eps_, v_, sc1start, sc1end, p.sc1);
    get_MPO_bath(ampo, eps_, v_, sc2start, sc2end, p.sc2);
    

    H = toMPO(ampo);

}
