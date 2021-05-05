#ifndef _MPSALGS_H
#define _MPSALGS_H

inline void
  plussers(Index const& l1,
	   Index const& l2,
	   Index      & sumind,
	   ITensor    & first,
	   ITensor    & second)
{
  if(not hasQNs(l1) && not hasQNs(l2))
    {
      auto m = dim(l1)+dim(l2);
      if(m <= 0) m = 1;
      sumind = Index(m,tags(sumind));

      first = delta(l1,sumind);
      auto S = Matrix(dim(l2),dim(sumind));
      for(auto i : range(dim(l2)))
	{
	  S(i,dim(l1)+i) = 1;
	}
      second = matrixITensor(std::move(S),l2,sumind);
    }
  else
    {
      auto siq = stdx::reserve_vector<QNInt>(nblock(l1)+nblock(l2));
      for(auto n : range1(nblock(l1)))
	{
	  siq.emplace_back(qn(l1,n),blocksize(l1,n));
	}
      for(auto n : range1(nblock(l2)))
	{
	  siq.emplace_back(qn(l2,n),blocksize(l2,n));
	}
#ifdef DEBUG
      if(siq.empty()) Error("siq is empty in plussers");
#endif
      sumind = Index(std::move(siq),
		     dir(sumind),
		     tags(sumind));
      first = ITensor(dag(l1),sumind);
      int n = 1;
      for(auto j : range1(nblock(l1)))
	{
	  auto D = Tensor(blocksize(l1,j),blocksize(sumind,n));
	  auto minsize = std::min(D.extent(0),D.extent(1));
	  for(auto i : range(minsize)) D(i,i) = 1.0;
	  getBlock<Real>(first,{j,n}) &= D;
	  ++n;
	}
      second = ITensor(dag(l2),sumind);
      for(auto j : range1(nblock(l2)))
	{
	  auto D = Tensor(blocksize(l2,j),blocksize(sumind,n));
	  auto minsize = std::min(D.extent(0),D.extent(1));
	  for(auto i : range(minsize)) D(i,i) = 1.0;
	  getBlock<Real>(second,{j,n}) &= D;
	  ++n;
	}
    }
}

#endif
