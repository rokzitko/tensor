include /home/zitko/repos/itensor/options.mk

ITENSOR_LIBDIR=${ITENSOR}/lib
PREFIX=${ITENSOR}

TENSOR_HEADERS=$(PREFIX)/itensor/all.h
CCFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(CPPFLAGS) $(OPTIMIZATIONS) -fopenmp
CCGFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(DEBUGFLAGS) -fopenmp
LIBFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBFLAGS)
LIBGFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBGFLAGS)

#Rules ------------------

%.o: %.cc $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

FindGS.o: FindGS.cc $(ITENSOR_LIBS) $(TENSOR_HEADERS) SC_BathMPO.h
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

FindGS-middle.o: FindGS.cc $(ITENSOR_LIBS) $(TENSOR_HEADERS) SC_BathMPO_MiddleImp.h
	$(CCCOM) -DMIDDLE_IMP -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

# calcGS targets -----------------

calcGStargets: calcGS calcGS-middle

calcGS: calcGS.o FindGS.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) calcGS.o FindGS.o -o calcGS $(LIBFLAGS)

calcGS-middle: calcGS.o FindGS-middle.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) calcGS.o FindGS-middle.o -o calcGS-middle $(LIBFLAGS)

# calcPT targets -----------------

calcPTtargets: calcPT calcPT-middle

calcPT: calcPT.o FindGS.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) calcPT.o FindGS.o -o calcPT $(LIBFLAGS)

calcPT-middle: calcPT.o FindGS-middle.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) calcPT.o FindGS-middle.o -o calcPT-middle $(LIBFLAGS)


buildPT: calcPTtargets
buildGS: calcGStargets
build: calcGStargets calcPTtargets
