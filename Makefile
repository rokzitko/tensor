include ${ITENSOR}/options.mk

ITENSOR_LIBDIR=${ITENSOR}/lib
PREFIX=${ITENSOR}

TENSOR_HEADERS=$(PREFIX)/itensor/all.h
CCFLAGS= -I. -I./include $(ITENSOR_INCLUDEFLAGS) $(CPPFLAGS) $(OPTIMIZATIONS) -Wno-unused-variable -fconcepts
CCGFLAGS= -I. -I./include $(ITENSOR_INCLUDEFLAGS) $(DEBUGFLAGS) -Wno-unused-variable -fconcepts
LIBFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBFLAGS) -lhdf5 -ltbb $(myRPATH)
LIBGFLAGS=-L'$(ITENSOR_LIBDIR)' $(ITENSOR_LIBGFLAGS) -lhdf5 -ltbb -Wl,-rpath,$(myRPATH)
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

calcGS-g: mkdebugdir .debug_objs/calcGS.o .debug_objs/FindGS.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/calcGS.o .debug_objs/FindGS.o -o calcGS-g $(LIBGFLAGS)

# calcOverlap targets -----------------

calcOverlap: calcOverlap.o FindGS.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) calcOverlap.o FindGS.o -o calcOverlap $(LIBFLAGS)

calcOverlap-g: mkdebugdir .debug_objs/calcOverlap.o .debug_objs/FindGS.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/calcOverlap.o .debug_objs/FindGS.o -o calcOverlap-g $(LIBGFLAGS)

# MF_selfConsistent_calcGS targets -----------------

MF_selfConsistent_calcGS: MF_selfConsistent_calcGS.o FindGS.o $(ITENSOR_LIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCFLAGS) MF_selfConsistent_calcGS.o FindGS.o -o MF_selfConsistent_calcGS $(LIBFLAGS)

MF_selfConsistent_calcGS-g: mkdebugdir .debug_objs/MF_selfConsistent_calcGS.o .debug_objs/FindGS.o $(ITENSOR_GLIBS) $(TENSOR_HEADERS)
	$(CCCOM) $(CCGFLAGS) .debug_objs/MF_selfConsistent_calcGS.o .debug_objs/FindGS.o -o MF_selfConsistent_calcGS-g $(LIBGFLAGS)


buildGS: calcGS
buildOverlap: calcOverlap
buildMF_selfConsistent_calcGS: MF_selfConsistent_calcGS
build: calcGS calcOverlap MF_selfConsistent_calcGS
debug: calcGS-g

clean:
	rm -v *.o calcGS calcPT calcOverlap

mkdebugdir:
	mkdir -p .debug_objs
