#include <vector>
using namespace itensor;


//fills the MPO tensors
void Fill_SCBath_MPO(MPO& H, const SiteSet& sites_, const std::vector<double>& eps_, 
                const std::vector<double>& v_, double U_, double Vbar_ )
{
      //QN objects are necessary to have abelian symmetries in MPS
      QN    qn0 ( {"Sz",  0},{"Nf", 0,-1} ),
            cup ( {"Sz", +1},{"Nf",+1,-1} ),
            cdn ( {"Sz", -1},{"Nf",+1,-1} ),  
            cupD( {"Sz", -1},{"Nf",-1,-1} ), 
            cdnD( {"Sz", +1},{"Nf",-1,-1} ); 


    std::vector<Index> links;
    links.push_back( Index() );

    //first we create the link indices which carry quantum number information
    for(auto i : range1(length(H))){
        links.push_back(Index(  qn0,  2,
                                cup,  1,
                                cdn,  1,
                                cupD, 1,
                                cdnD, 1,
                                cup+cdn,   1,
                                cupD+cdnD, 1,     Out, "Link" ));
        
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

        W = ITensor(right, sites_.si(i), sites_.siP(i) );
        W += sites_.op("Id",i) * setElt(right(1));

        W += sites_.op("Ntot",i)  * setElt(right(2)) * eps_[i-1];
        W += sites_.op("Nupdn",i) * setElt(right(2)) * U_;

        W += sites_.op("Cup",i) * setElt(right(3));
        W += sites_.op("Cdn",i) * setElt(right(4));
        W += sites_.op("Cdagup",i) * setElt(right(5));
        W += sites_.op("Cdagdn",i) * setElt(right(6));

        //std::cout << "using " << eps_[i-1] <<std::endl;
    }

    // sites 2 ... N-1 are matrices
    for(auto i : range1(2,length(H)-1)){
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );
        Index right = links.at(i);

        W = ITensor(left, right, sites_.si(i), sites_.siP(i) );

        W += sites_.op("Id",i) * setElt(left(1), right(1));

        W += sites_.op("Ntot",i)           * setElt(left(1),right(2)) * eps_[i-1];
        W += sites_.op("Nupdn",i)          * setElt(left(1),right(2)) * (Vbar_);

        W += sites_.op("Cdn*Cup",i)        * setElt(left(1),right(7)) * Vbar_;
        W += sites_.op("Cdagup*Cdagdn",i)  * setElt(left(1),right(8)) * Vbar_;

        W += sites_.op("Id",i)*setElt(left(2),right(2));
        W += sites_.op("F" ,i)*setElt(left(3),right(3));
        W += sites_.op("F" ,i)*setElt(left(4),right(4));
        W += sites_.op("F" ,i)*setElt(left(5),right(5));
        W += sites_.op("F" ,i)*setElt(left(6),right(6));
        W += sites_.op("Id",i)*setElt(left(7),right(7));
        W += sites_.op("Id",i)*setElt(left(8),right(8));

        W += sites_.op("Cdagup",i)*setElt(left(3),right(2))* v_[i-1];
        W += sites_.op("Cdagdn",i)*setElt(left(4),right(2))* v_[i-1];
        W += sites_.op("Cup",   i)*setElt(left(5),right(2))* v_[i-1];
        W += sites_.op("Cdn",   i)*setElt(left(6),right(2))* v_[i-1];

        W += sites_.op("Cdagup*Cdagdn",i)*setElt(left(7),right(2));
        W += sites_.op("Cdn*Cup",i)      *setElt(left(8),right(2));

        //std::cout << "using " << eps_[i-1] << " and "<<v_[i-1]<<std::endl;
    }

    //site N is a vector again
    {
        int i = length(H);
        ITensor& W = H.ref(i);
        Index left = dag( links.at(i-1) );

        W = ITensor(left, sites_.si(i), sites_.siP(i) );
        
        W += sites_.op("Ntot",  i) * setElt(left(1)) * eps_[i-1];
        W += sites_.op("Nupdn",i)  * setElt(left(1)) * (Vbar_);

        W += sites_.op("Id",    i) * setElt(left(2)) ;
        W += sites_.op("Cdagup",i) * setElt(left(3)) * v_[i-1];
        W += sites_.op("Cdagdn",i) * setElt(left(4)) * v_[i-1];
        W += sites_.op("Cup",   i) * setElt(left(5)) * v_[i-1];
        W += sites_.op("Cdn",   i) * setElt(left(6)) * v_[i-1];

        W += sites_.op("Cdagup*Cdagdn",i) * setElt(left(7));
        W += sites_.op("Cdn*Cup",      i) * setElt(left(8));

        //std::cout << "using " << eps_[i-1] << " and "<<v_[i-1]<<std::endl;
    }   
    
  }

