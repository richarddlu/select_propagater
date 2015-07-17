LIBS = \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgproc

all:	demo

demo:	demo.o select_propagater.o nnls.o
		g++ demo.o select_propagater.o nnls.o -o demo $(LIBS)

demo.o:	demo.cpp
		g++ -c demo.cpp

select_propagater.o:	select_propagater.cpp
						g++ -c select_propagater.cpp

nnls.o:	nnls.cpp
		g++ -c nnls.cpp

clean:
	rm -f demo demo.o select_propagater.o nnls.o