# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/axtyax/Projects/LightNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/axtyax/Projects/LightNN/build

# Include any dependencies generated for this target.
include CMakeFiles/LightNN.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LightNN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LightNN.dir/flags.make

CMakeFiles/LightNN.dir/include/network.o: CMakeFiles/LightNN.dir/flags.make
CMakeFiles/LightNN.dir/include/network.o: ../include/network.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/axtyax/Projects/LightNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LightNN.dir/include/network.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LightNN.dir/include/network.o -c /home/axtyax/Projects/LightNN/include/network.cpp

CMakeFiles/LightNN.dir/include/network.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LightNN.dir/include/network.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/axtyax/Projects/LightNN/include/network.cpp > CMakeFiles/LightNN.dir/include/network.i

CMakeFiles/LightNN.dir/include/network.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LightNN.dir/include/network.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/axtyax/Projects/LightNN/include/network.cpp -o CMakeFiles/LightNN.dir/include/network.s

CMakeFiles/LightNN.dir/include/network.o.requires:

.PHONY : CMakeFiles/LightNN.dir/include/network.o.requires

CMakeFiles/LightNN.dir/include/network.o.provides: CMakeFiles/LightNN.dir/include/network.o.requires
	$(MAKE) -f CMakeFiles/LightNN.dir/build.make CMakeFiles/LightNN.dir/include/network.o.provides.build
.PHONY : CMakeFiles/LightNN.dir/include/network.o.provides

CMakeFiles/LightNN.dir/include/network.o.provides.build: CMakeFiles/LightNN.dir/include/network.o


# Object files for target LightNN
LightNN_OBJECTS = \
"CMakeFiles/LightNN.dir/include/network.o"

# External object files for target LightNN
LightNN_EXTERNAL_OBJECTS =

libLightNN.so: CMakeFiles/LightNN.dir/include/network.o
libLightNN.so: CMakeFiles/LightNN.dir/build.make
libLightNN.so: CMakeFiles/LightNN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/axtyax/Projects/LightNN/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libLightNN.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LightNN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LightNN.dir/build: libLightNN.so

.PHONY : CMakeFiles/LightNN.dir/build

CMakeFiles/LightNN.dir/requires: CMakeFiles/LightNN.dir/include/network.o.requires

.PHONY : CMakeFiles/LightNN.dir/requires

CMakeFiles/LightNN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LightNN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LightNN.dir/clean

CMakeFiles/LightNN.dir/depend:
	cd /home/axtyax/Projects/LightNN/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/axtyax/Projects/LightNN /home/axtyax/Projects/LightNN /home/axtyax/Projects/LightNN/build /home/axtyax/Projects/LightNN/build /home/axtyax/Projects/LightNN/build/CMakeFiles/LightNN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/LightNN.dir/depend

