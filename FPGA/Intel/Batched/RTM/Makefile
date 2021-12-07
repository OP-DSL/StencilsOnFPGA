all:
	g++ -c populate_cpu.cpp -o populate_cpu.o
	g++ -c rtm_app.cpp -o rtm_app.o
	g++ -o rtm_cpu populate_cpu.o rtm_app.o

clean:
	rm  rtm_cpu *.o