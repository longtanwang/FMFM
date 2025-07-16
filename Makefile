# Compiler and flags
FC = gfortran
FFLAGS = -O -fPIE -Isrc
LDFLAGS = -pie

# Directories
BINDIR = bin
BUILDDIR = build
SRCDIR = src

# Source files and object files
SRCS = $(wildcard $(SRCDIR)/*.f)
OBJS = $(patsubst $(SRCDIR)/%.f, $(BUILDDIR)/%.o, $(SRCS))
# Specify only the object files needed for hash_driver
OBJS3 = $(BUILDDIR)/fmamp_subs.o \
        $(BUILDDIR)/pol_subs.o \
        $(BUILDDIR)/station_subs_5char.o \
        $(BUILDDIR)/uncert_subs.o \
        $(BUILDDIR)/util_subs.o \
        $(BUILDDIR)/vel_subs.o

# Executable name
EXEC = $(BINDIR)/hash_driver

# Default rule
all: $(EXEC)

# Rule to link the executable
$(EXEC): $(OBJS3) $(BUILDDIR)/hash_driver.o
	mkdir -p $(BINDIR)
	$(FC) $(FFLAGS) $(LDFLAGS) -o $@ $^

# Rules to compile source files
$(BUILDDIR)/%.o: $(SRCDIR)/%.f
	mkdir -p $(BUILDDIR)
	$(FC) $(FFLAGS) -c -o $@ $<

# Clean rule
clean:
	rm -f $(BUILDDIR)/*.o $(EXEC)

.PHONY: all clean
