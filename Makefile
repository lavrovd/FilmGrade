CUDAPATH = /usr/local/cuda
NVCC = ${CUDAPATH}/bin/nvcc
CXXFLAGS = -I../OpenFX-1.3/include -I../Support/include
OPTIMIZER = -g

FilmGradePlugin.ofx: FilmGradePlugin.o CudaKernel.o OpenCLKernel.o ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) -bundle $^ -o $@ -L${CUDAPATH}/lib -lcuda -lcudart -F/Library/Frameworks -framework CUDA -framework OpenCL
	mkdir -p FilmGradePlugin.ofx.bundle/Contents/MacOS/
	cp FilmGradePlugin.ofx FilmGradePlugin.ofx.bundle/Contents/MacOS/
	mkdir -p FilmGradePlugin.ofx.bundle/Contents/Resources/
	cp OpenFX.Yo.FilmGrade.png FilmGradePlugin.ofx.bundle/Contents/Resources/
	rm *.o *.ofx

CudaKernel.o: CudaKernel.cu
	${NVCC} -c $<

%.o: ../Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)
	
clean:
	rm -f *.o *.ofx
	rm -fr FilmGradePlugin.ofx.bundle
