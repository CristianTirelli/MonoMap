# Monomorphism-based CGRA Mapping via Space and Time Decoupling


### Abstract:

Coarse-Grain Reconfigurable Arrays (CGRAs) provide flexibility and energy efficiency in accelerating compute-intensive loops. Existing compilation techniques often struggle with scalability, unable to map code onto large CGRAs. To address this, we propose a novel approach to the mapping problem where the time and space dimensions are decoupled and explored separately. We leverage a SMT formulation to traverse the time dimension first, and then perform a monomorphism-based search to find a valid spatial solution. Experimental results show that our approach achieves the same mapping quality of state of the art techniques while significantly reducing compilation time, with this reduction being particularly tangible when compiling for large CGRAs. We achieve approximately $10^5\times$ average compilation speedup for the benchmarks evaluated on a `20x20` CGRA.

## Requirements 
This project was developed and tested on `Ubuntu 20.04.6`. \
To generate the CMake files, we used `CMake` version `3.23` and to compile  the project, we used `ninja` version `1.10.0`. \

## First start:
1. Run `setup.sh` to compile LLVM and configure the Python environment.
2) Before using the compiler, activate the virtual environment with:
``` bash
source mono-compiler/bin/activate
```



## Supported Code
In the `benchmarks` folder, you'll find sample code that can be used to map onto a CGRA. 
To compile your own code, simply add  ```#pragma cgra acc``` before the loop you'd like to map to the CGRA.

**Note**: Currently, the compiler only supports:
- Innermost loops.
- Loops without function calls or conditionals.

#### Example (from `benchmarks/reverse_bits`):
```c
#pragma cgra acc
for (int i = rev = 0; i < NumBits; i++)
{
    rev = (rev << 1) | (index & 1);
    index >>= 1;
}
```



## Compilation Instructions
After adding the ```#pragma cgra acc```  directive to your code, compile it with the following command:

```bash
./monolang -f benchmarks/sha2/sha.c
```

By default, the code will be compiled for a 4x4 CGRA. To specify a different CGRA size, use the `-x` and `-y` options. For example, to compile for a 5x5 CGRA, run:

```bash
./cgralang -f benchmarks/sha2/sha.c -x 5 -y 5
```
## Output
The output of the compiler is a file called `cgra-mono-code-acc`, which includes various debug information and the mapping result.


