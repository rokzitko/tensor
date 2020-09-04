include /home/zitko/repos/itensor/options.mk

ITENSOR_LIBDIR=${ITENSOR}/lib
PREFIX=${ITENSOR}

TENSOR_HEADERS=$(PREFIX)/itensor/all.h
CCFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(CPPFLAGS) $(OPTIMIZATIONS) -fopenmp -Wno-unused-variable
CCGFLAGS= -I. $(ITENSOR_INCLUDEFLAGS) $(DEBUGFLAGS) -fopenmp
LIBFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBFLAGS)
LIBGFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBGFLAGS)
MPOFILES = $(wildcard SC_BathMPO*.h)

#Rules ------------------

%.o: %.cc $(ITENSOR_LIBS) $(TENSOR_HEADERS) FindGS.h
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

FindGS.o: FindGS.cc FindGS.h $(ITENSOR_LIBS) $(TENSOR_HEADERS) $(MPOFILES)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

# calcGS targets -----------------

calcGS: calcGS.o FindGS.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) calcGS.o FindGS.o -o calcGS $(LIBFLAGS)

# calcPT targets -----------------

calcPT: calcPT.o FindGS.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) calcPT.o FindGS.o -o calcPT $(LIBFLAGS)

buildPT: calcPT
buildGS: calcGS
build: calcGS

clean:
	rm -v *.o calcGS calcPT
