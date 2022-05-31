BINARY=bin.out
CODEDIRS=. lib lib/utils
INCDIRS=. ./include/ ./include/utils ./include/enums

LINKDIR=

CPP=g++

#generates files that encode make rules for the .h dependencies
DEPFLAGS=-MP -MD

#automatically add the -I to each include directory
CPPFLAGS=-Wall -g $(foreach D,$(INCDIRS),-I$(D)) $(DEPFLAGS)

CPPFILES=$(foreach D,$(CODEDIRS),$(wildcard $(D)/*.cpp))
OBJECTS=$(patsubst %.cpp,%.o,$(CPPFILES))
DEPFILES=$(patsubst %.cpp,%.d,$(CPPFILES))

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CPP) -o $@ $^ $(LINKDIR)

%.o:%.cpp
	$(CPP) $(CPPFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJECTS) $(BINARY) $(DEPFILES)

-include $(DEPFILES)