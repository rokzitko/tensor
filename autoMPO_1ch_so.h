inline void get_autoMPO_1ch_so(MPO& H, const double Eshift, const std::vector<double>& eps_,
                const std::vector<double>& v_, const params &p)
{
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

    // bath terms     
    for(auto i : range1(2, p.N)){
        ampo += eps_[i-1],"Ntot",i;
        for(auto j: range1(2, p.N)){
            ampo += p.sc->g(j-1),"Cdagup",i,"Cdagdn",i,"Cdn",j,"Cup",j;
        }
    }

    
    //hopping terms
    for(auto i : range1(2, p.N-1)){
        ampo += p.sc->t(),"Cdagup",i,"Cup",i+1;
        ampo += p.sc->t(),"Cdagdn",i,"Cdn",i+1;
        ampo += p.sc->t(),"Cdagup",i+1,"Cup",i;
        ampo += p.sc->t(),"Cdagdn",i+1,"Cdn",i;
    }

    //spin orbit coupling terms
    for (auto i : range(2, p.N)){
        for (auto j : range(2, p.N)){
            if (i != j){

                auto so_prefactor = Complex_i * p.sc->l();
                so_prefactor *=  i < j ? 1 : -1;

                ampo += so_prefactor,"Cdagup",i,"Cdn",j;
                ampo += so_prefactor,"Cdagdn",i,"Cup",j;
            }
        }
    }


    H = toMPO(ampo);

}
