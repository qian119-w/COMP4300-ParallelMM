all : gen_PartA gen_PartB

gen_PartA : gen_PartA.c
	gcc -g -Wall -o $@ $? 

gen_PartB : gen_PartB.c
	gcc -g -Wall -o $@ $? 

clean : 
	/bin/rm -f *.o *~ gen_PartA gen_PartB