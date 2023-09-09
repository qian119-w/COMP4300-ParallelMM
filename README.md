# COMP4300 Individual Project

In the individual project, implement the parallel matrix multiplication algorithms covered by the lecture slides with parallel programming APIs. There are two parts:

## Part A. Basic Tasks (75% of total project marks)

1. Implement a parallel version of blocked matrix multiplication by OpenMP (10%).
2. Implement SUMMA algorithm by MPI (20%).
3. Implement Cannon's algorithm by MPI (20%).
4. Implement SUMMA and Cannon's algorithms by Pthreads. You may implement the communications among threads by various approaches (e.g., mutexes, semaphores) (25%).

The code for Tasks 1-4 should be able to accept external testcase input files. For example, if Task1 is the executable file, it should accept following command line:
```
Task1 <num_threads> <input_testcase_filename> <output_filename>
```
See the code/testcase folder for the format description and examples of input and output files. You should also provide your own testcases, together with suitable descriptions in md files.

For each task, you are required to provide at least two variants of implementation. One is expected to improve over the other. Compare and benchmark the performance of the variants in the following aspects:
- Measure the running times, speed-ups and efficiencies 
- Compare with different dimensions of matrices, and sub-matrix block sizes 
- Compare with various numbers of threads/processes 
- Compare the performance on more than one platform


### Deliverables:
-	C programing code implementing Tasks 1-4 with detailed and clear inline comments and documentation (in md file), and relevant make files. The code should comply with the input and output file formats. 

-	Detailed report of measured performance with suitable data plots, discussions of results (e.g., strong/weak scalability), and explanations of the performance differences (e.g., on different platforms, or variants of implementation). The report may be written in md or pdf files.


## Part B. Open Tasks (25% of total project marks)
In addition to the basic tasks, you are encouraged to implement more advanced tasks, such as
-	Parallel Strassen algorithm
-	CUDA parallel matrix multiplication (blocked multiplication, SUMMA, Cannon's algorithms)
-	Other parallel algorithms for linear algebras (e.g., solving linear equations, spectral decomposition)

The required deliverables for Part B are similar to those of Part A.



## Marking Criteria
The deliverables will be evaluated holistically based on the following criteria:
-   Performance of implementation
-   Technical difficulty of open tasks
-	Correctness and quality of code
-	Coverage of performance measurements and benchmarks
-	Insightfulness of report 
-	Readability of documentation

Note that the report and code will be considered equally important. But we will evaluate the report and code together, because they are interrelated (e.g., the measurements and benchmarks in the report reflect the performance of implementation.)  

The default marking environment is Gadi or (GPU: stugpu2.anu.edu.au). 

## Plagiarism 
We are aware that there is some sample code available on the Internet. But you are strongly encouraged to implement your answers on your own. We will run extensive plagiarism checks, comparing with various online sources. Heavy penalty will be imposed for any detected plagiarism.

## Submission Guidelines 

Fork the project repo, then add the marker user `comp4300-2022-s1-marker` as a developer to your repo. Commit and push your code and report to your repo before submission deadline. We will mark your answers and provide feedbacks to your repo.

# Submission Deadline
You must commit and push your code to your repo before **Friday of Week 12**. Any late updates after the deadline will not be considered.  
