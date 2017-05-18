UNAME_SYSTEM := $(shell uname -s)

CUDAPATH = /usr/local/cuda
NVCC = ${CUDAPATH}/bin/nvcc
CXXFLAGS = -I../OpenFX-1.3/include -I../Support/include

ifeq ($(UNAME_SYSTEM), Linux)
	OPENCLPATH = /opt/AMDAPP
	CXXFLAGS += -I${CUDAPATH}/include -I${OPENCLPATH}/include -fPIC
	NVCCFLAGS = --compiler-options="-fPIC"
	LDFLAGS = -shared -L${CUDAPATH}/lib64 -lcuda -lcudart
	BUNDLE_DIR = FilmGradePlugin.ofx.bundle/Contents/Linux-x86-64/
	BUNDLE_RES = FilmGradePlugin.ofx.bundle/Contents/Resources/
else
	LDFLAGS = -bundle -L${CUDAPATH}/lib -lcuda -lcudart -F/Library/Frameworks -framework CUDA -framework OpenCL
	BUNDLE_DIR = FilmGradePlugin.ofx.bundle/Contents/MacOS/
	BUNDLE_RES = FilmGradePlugin.ofx.bundle/Contents/Resources/
	
endif

FilmGradePlugin.ofx: FilmGradePlugin.o CudaKernel.o OpenCLKernel.o ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp FilmGradePlugin.ofx $(BUNDLE_DIR)
	mkdir -p $(BUNDLE_RES)
	cp OpenFX.Yo.FilmGrade.png $(BUNDLE_RES)
	rm *.o *.ofx

CudaKernel.o: CudaKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

%.o: ../Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)
	
clean:
	rm -f *.o *.ofx
	rm -fr FilmGradePlugin.ofx.bundle
