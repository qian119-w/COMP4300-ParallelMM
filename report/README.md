# Instructions
Makefiles are provided for compilation. Hence, to compile the source files under directory `PartA` and `PartB`, use the command:
```
make <file_name>
```

To run an executable `file_name`, use the following commands:
  - **OpenMP** & **Pthreads**:
    ```
    ./file_name <num_threads> <input_filename>
    ```
  - **MPI**:
    ```
    mpiexec -n <num_proc> ./file_name <input_filename>
    ```
  - **CUDA** (PartB): 
    ```
    ./cuda_sort <num_threads> <input_filename> 
    ``` 
    ```
    ./cuda_sort_2 <num_blks> <num_thds_per_blk> <input_filename>
    ```




## Important Note:
The output style of the programs differs from the project specifications. The implementations do not write the result and timing of matrix multiplication (**PartA**) and sample sort (**PartB**) in a separate output file. I find it more straightforward to verify the correctness of results with an automated tester. The tester functions are provided in `utils.c` under the directory `util`. Testers are already included at the end of each implementation. If the results are correct, the tester will output the string `'Mat Mul/Sorting correct'` in the terminal; otherwise `'Mat Mul/Sorting incorrect'` is written.
Timings of the implementations will also be displayed in the terminal.

Testcase generators `gen_PartA.c` and `gen_PartB.c` are provided under the directory `testcases`. The generators generate testcases in textfiles `testcase_xxx` and provide the expected results in textfiles `output_xxx` for automatic testing. Serial matrix multiplication algorithm (**PartA**) and the native `qsort` function (**PartB**) are used to produce the expected results. 

Makefile is provided for compilation. To compile the generators, use the command:
```
make <file_name>
```

To run the executable `gen_PartA` or `gen_PartB`, use the command:
```
./gen_Part(A/B)
```
User will be prompted to enter matrix dimensions as 3 integers `<M> <K> <N>` for `gen_PartA`, and array length as 1 integer `<n>` for `gen_PartB`.

