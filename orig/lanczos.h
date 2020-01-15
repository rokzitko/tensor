#ifndef __FORKLANZCOS__
#define __FORKLANZCOS__

#include "itensor/itensor.h"
#include "itensor/util/print_macro.h"


using namespace itensor;

namespace forktps {

    /** \file lanczos.h
    * In this file we implement lanczos functions used by DMRG or TDVP
    */


    void ev_hermitian(double *MAT, double *diag, int L, char JOBZ) {
      int LDA = L, info, LWORK = 2 * L * L;
      double *WORK = new double[LWORK];
      char UPLO = 'U';

      dsyev_(&JOBZ, &UPLO, &L, MAT, &LDA, diag, WORK, &LWORK, &info);

      delete[] WORK;
      return;
    }

    void FillHtri( double* Htri, int L, std::vector<double> eps, std::vector<double> kappa ){
      for (auto k : range(L*L))
        Htri[k] = 0;
      

      

      for (auto k : range(L)) {
        Htri[k * (L + 1)] = eps[k];
        if (k != 0) {
          Htri[k * L + k - 1] = kappa[k];
        } // add upper
        if (k != L - 1) {
          Htri[k * L + k + 1] = kappa[k + 1];
        } // add lower tridiag
      }
    }

    /**
    * Calculates the ground state of matrix \p H with the lanzcos procedure using
    * \p phi as a starting vector. It accepts any matrix, as long as H.product( ...
    * ) is defined. \p args uses the following parameters
    * @param     NLanzcosMax: Maximum number of Lanzcos iterations
    * @param     LanzcosStep: Number of Krylov vectors created, before the
    * tridiagonal eigenvalue problem is solved again
    * @param     LanzcosConv: Energy convergence criteria
    */
    template <class Matrix>
    double lanzcos(const Matrix& H, ITensor &phi, Args &args = Args::global()) {

      // args can change the following parameters:

      int NKrylovMax = args.getInt("NKrylovMax", 50),
          DiagEvery = args.getInt("DiagEvery", 4), StepsDone = 0;
      double energy = 0, energyLastIt = 0, 
            KrylovNormError = args.getReal("KrylovNormError", 1E-13),
            energyConv = args.getReal("LanzcosConv", 1E-12);

      bool basisFull = false;

      std::vector<double> eps, kappa, GS;
      eps.resize(0);
      kappa.resize(0);
      std::vector<ITensor> KrylovVecs;

      if(H.size() < NKrylovMax )
        NKrylovMax = H.size();

      ITensor xn, xnm1, xnp1;
      xn = phi;
      xn /= norm(xn);

      for (auto i :range(NKrylovMax)){ 
        kappa.push_back(norm(xn));

        if (std::fabs(kappa.back()) > KrylovNormError) 
          xn /= kappa.back();
        else 
          basisFull = true;
        
        KrylovVecs.push_back(xn);

        H.product(xn, xnp1);
        ITensor ovlp = xn * dag(xnp1);
        eps.push_back(std::real(ovlp.cplx()));

        xnp1 += -eps[i] * xn; 
        if(i>0)
          xnp1 += - kappa[i] * xnm1;

        xnm1 = xn;
        xn = xnp1;

        //diagonalize matrix if any of the three conditions is fullfilled
        // 1. automatically diagonalize ever DiagEvery step
        // 2. basis is full, i.e.: norm(kappa) < KrylovNormError
        // 3. maximum number of krylov vectors is reached
        if ((i % DiagEvery == 0 && i != 0) || basisFull || i == NKrylovMax-1 ) {
          StepsDone = basisFull ? i : i+1;

          double *Htri = new double[StepsDone * StepsDone];
          double *diag = new double[StepsDone];

          FillHtri(Htri, StepsDone, eps, kappa);

          ev_hermitian(Htri, diag, StepsDone, 'V');
        
          energy = diag[0];
          GS.resize(StepsDone);
          for (auto k : range(StepsDone)) 
            GS[k] = Htri[k];
          
          delete[] Htri;
          delete[] diag;

          // converged?
          if (std::fabs(energy - energyLastIt) < energyConv || basisFull) 
            break;

          energyLastIt = energy;
        }

      }

      args.add("LanzcosStepsDone", StepsDone);

      // xn=phi; xnm1 = ITensor(), xnp1 = ITensor();
      for (auto i : range(StepsDone)) {
        i == 0 ? phi = GS.at(i)*KrylovVecs.at(i) : phi += GS.at(i)*KrylovVecs.at(i);

        // xn /= kappa[i];
        //H.product(xn,xnp1);
        //xnp1 += -eps[i]*xn - kappa[i]*xnm1;
        //xnm1 = xn;
        //xn=xnp1;
        //xn.scaleTo(1); 
      }

      //phi.scaleTo(1);

      return energy;
    }






} // end of namespace itensor

#endif
