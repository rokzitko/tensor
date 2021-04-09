inline void makeS2_MPO(MPO& S2, const params &p)
{

    auto ampo = AutoMPO(p.sites);

    for (auto i : range1(p.N)){
        for (auto j : range1(p.N)){

            ampo += 1.,"Sz",i,"Sz",j;

            ampo += 0.5,"S+",i,"S-",j;
        
            ampo += 0.5,"S-",i,"S+",j;
        }
    }

    S2 = toMPO(ampo);

}
