

// Luka, Sep 2023
// Below is the extension of the LocalMPO_MPS class from itensor/mps/localmpo.h that takes in a vector of MPOs AND a vector of MPS.
// The built-in methods can only do one or the other.

// IDEA: 
// for enforcing spin, I want to pass both a vector of MPOs (H and S^2 operators) and a vector of MPSs (previously found states, to 
// compute excited states) to the dmrg() function. 
// The existing options only allow passing either a vector of MPOs OR a vector of MPSs. In dmrg(), these are effectively summed up into 
// a local object (named PH). 
// If a vector of MPOs is passed, this is an instance of LocalMPOSet (class defined in itensor/mps/localmposet.h) and 
// if a vector of MPSs, an instance of LocalMPOS_MPS (class defined in itensor/mps/localmpo_mps.h). 
// This is then passed to DMRGWorker, where an iterative diagonalisation function is called, davidson(..., PH, ...). It works if 
// the object PH has methods product(), size() and diag().
//
// Here, take a class that inherits LocalMPOS_MPS


#ifndef __myclass
#define __myclass
#include "itensor/mps/localmpo.h"

namespace itensor {

// This is a copy of the LocalMPOSet class from itensor/mps/localmposet.h,
// extended to carry a vector of MPS as well! 
class local_MPO_and_MPS
    {
    std::vector<MPO> const* Op_ = nullptr;
    std::vector<LocalMPO> lmpo_; // LocalMPO objects representing projected version of the MPO in Op_
    std::vector<LocalMPO> lmps_; // LocalMPO objects representing projected version of MPS in psis_
    Real weight_ = 1; // weight for the MPSs
    public:

    local_MPO_and_MPS(Args const& args = Args::global()) { }

    local_MPO_and_MPS(
        std::vector<MPO> const& Op,
        std::vector<MPS> const& psis,
        Args const& args = Args::global());

    local_MPO_and_MPS( 
        std::vector<MPO> const& H, 
        std::vector<ITensor> const& LH, 
        int LHlim,
        std::vector<ITensor> const& RH,
        int RHlim,
        std::vector<MPS> const& psis,
        std::vector<ITensor> const& Lpsi,
        std::vector<ITensor> const& Rpsi,
        Args const& args = Args::global());

    void
    product(ITensor const& phi, 
            ITensor & phip) const;

    Real
    expect(ITensor const& phi) const;

    ITensor
    deltaRho(ITensor const& AA, 
             ITensor const& comb, 
             Direction dir) const;

    ITensor
    diag() const;

    void
    position(int b, MPS const& psi);

    std::vector<ITensor>
    L() const 
        { 
        auto L = std::vector<ITensor>(lmpo_.size());
        for(auto n : range(lmpo_)) L.at(n) = lmpo_.at(n).L();
        return L;
        }
    std::vector<ITensor>
    R() const 
        { 
        auto R = std::vector<ITensor>(lmpo_.size());
        for(auto n : range(lmpo_)) R.at(n) = lmpo_.at(n).R();
        return R;
        }

    void
    L(std::vector<ITensor> const& nL)
        { 
        for(auto n : range(lmpo_)) lmpo_.at(n).L(nL.at(n));
        }
    void
    R(std::vector<ITensor> const& nR)
        { 
        for(auto n : range(lmpo_)) lmpo_.at(n).R(nR.at(n));
        }

    void
    shift(int j, Direction dir, ITensor const& A)
        {
        for(auto n : range(lmpo_)) lmpo_[n].shift(j,dir,A);
        }

    int
    numCenter() const { return lmpo_.front().numCenter(); }
    void
    numCenter(int val);

    size_t
    size() const { return lmpo_.front().size(); }

    explicit
    operator bool() const { return bool(Op_); }

    Real
    weight() const { return weight_; }
    void
    weight(Real val) { weight_ = val; }

    bool
    doWrite() const { return lmpo_.front().doWrite(); }
    void
    doWrite(bool val, Args const& args = Args::global()) 
        { 
        for(auto& lm : lmpo_) lm.doWrite(val,args);
        }

    }; // end local_MPO_and_MPS

inline local_MPO_and_MPS::
local_MPO_and_MPS(
    std::vector<MPO> const& Op,
    std::vector<MPS> const& psis,
    Args const& args)
  : Op_(&Op),
    lmpo_(Op.size()),
    lmps_(psis.size()),
    weight_(args.getReal("Weight",1))
    { 
    for(auto n : range(lmpo_.size()))
        {
        lmpo_[n] = LocalMPO(Op.at(n),args);
        }
    for(auto j : range(lmps_.size()))
        {
        lmps_[j] = LocalMPO(psis[j],args);
        }
    }

inline local_MPO_and_MPS::
local_MPO_and_MPS(
    std::vector<MPO> const& H, 
    std::vector<ITensor> const& LH, 
    int LHlim,
    std::vector<ITensor> const& RH,
    int RHlim,
    std::vector<MPS> const& psis,
    std::vector<ITensor> const& Lpsi,
    std::vector<ITensor> const& Rpsi,
    Args const& args)
  : Op_(&H),
    lmpo_(H.size()),
    lmps_(psis.size()),
    weight_(args.getReal("Weight",1))
    { 
    for(auto n : range(lmpo_.size()))
        {
        lmpo_[n] = LocalMPO(H.at(n),LH.at(n),LHlim,RH.at(n),RHlim,args);
        }
    
    for(auto j : range(lmps_.size()))
        {
        lmps_[j] = LocalMPO(psis[j],Lpsi[j],Rpsi[j],args);
        }
    
    }

void inline local_MPO_and_MPS::
product(ITensor const& phi, 
        ITensor & phip) const
    {

    // Multiplies the set of local MPOs
    lmpo_.front().product(phi,phip);
    ITensor phi_n;
    for(auto n : range(1,lmpo_.size()))
        {
        lmpo_[n].product(phi,phi_n);
        phip += phi_n;
        }

    // Multiplies the set of local MPSs
    // Copied from LocalMPO_MPS::product, localmpo_mps.h, line 138.
    ITensor outer; 
    for(auto& M : lmps_)
        {
        M.product(phi,outer);
        outer *= weight_;
        phip += outer;
        }
    }

Real inline local_MPO_and_MPS::
expect(ITensor const& phi) const
    {
    Real ex_ = 0;
    for(size_t n = 0; n < lmpo_.size(); ++n)
    for(auto n : range(lmpo_.size()))
        {
        ex_ += lmpo_[n].expect(phi);
        }
    return ex_;
    }

ITensor inline local_MPO_and_MPS::
deltaRho(ITensor const& AA,
         ITensor const& comb, 
         Direction dir) const
    {
    ITensor delta = lmpo_.front().deltaRho(AA,comb,dir);
    for(auto n : range(1,lmpo_.size()))
        {
        delta += lmpo_[n].deltaRho(AA,comb,dir);
        }
    return delta;
    }

ITensor inline local_MPO_and_MPS::
diag() const
    {
    ITensor D = lmpo_.front().diag();
    for(auto n : range(1,lmpo_.size()))
        {
        D += lmpo_[n].diag();
        }
    return D;
    }

void inline local_MPO_and_MPS::
position(int b, 
         MPS const& psi)
    {
    for(auto n : range(lmpo_.size())){
        lmpo_[n].position(b,psi);
        }
    for(auto& M : lmps_){
        M.position(b,psi);
        }
    
    
    }

void inline local_MPO_and_MPS::
numCenter(int val)
    {
    for(auto n : range(lmpo_.size()))
        {
        lmpo_[n].numCenter(val);
        }
    }



// overload the dmrg call

Real inline
dmrg(
    MPS& psi,  // initial state
    std::vector<MPO> const& Hset,  // vector of MPOs
    std::vector<MPS> const& psis,  // vector of psis
    Sweeps const& sweeps, 
    Args const& args = Args::global())
    {
    if(hasQNs(psi))
        {
        auto psi_qn = totalQN(psi);
        for(auto n : range(psis))
            {
            auto qn_n = totalQN(psis[n]);
            if(qn_n != psi_qn)
                {
                printfln("totalQN of initial state:  %s",psi_qn);
                printfln("totalQN of state n=%d (n is 0-indexed): %s",n,qn_n);
                Error("Excited-state DMRG intended for states with same totalQN");
                }
            }
        }

    local_MPO_and_MPS PH (Hset, psis, args);
    Real energy = DMRGWorker(psi,PH,sweeps,args);
    return energy;
    } // dmrg


std::tuple<Real,MPS> inline
dmrg(
    std::vector<MPO> const& Hset,  // vector of MPOs
    std::vector<MPS> const& psis,  // vector of psis
    MPS const& psi0,               // initial guess for psi
    Sweeps const& sweeps, 
    Args const& args = Args::global())
    {
    auto psi = psi0;
    auto energy = dmrg(psi,Hset,psis,sweeps,args);
    return std::tuple<Real,MPS>(energy,psi);
    }

} //namespace itensor

#endif
