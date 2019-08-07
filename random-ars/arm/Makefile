CFLAGS=-g -O2 #-D CONFIG_ENABLE_PERIOD_64BIT
#CFLAGS=-g

all:
	gcc $(CFLAGS)  ars_v0.9.1.c

t: all
	./a.out | grep "bitmask=" | cut -d"=" -f2 | sh ./test_result.sh
