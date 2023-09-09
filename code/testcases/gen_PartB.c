#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int cmpfunc (const void * a, const void * b) {
  return ( *(int*)a - *(int*)b );
}

int main(int argc, char* argv[]){
  int n;
  printf("Input array length: <n>\n");
  scanf("%d", &n);
  char input_buf[50], output_buf[50];
  sprintf(input_buf, "testcase_sorting_%d", n);
  sprintf(output_buf, "output_sorting_%d", n);

  srand(time(NULL));
  FILE* fw = fopen(input_buf, "w");
  fprintf(fw, "%d\n", n);
  for (int i=0; i<n; i++){
    fprintf(fw, "%d ", rand());
    if (i > 0 && i % 100 == 0)
      fprintf(fw, "\n");
  }
  fclose(fw);

  FILE *fr = fopen(input_buf, "r");
  int * a;
  fscanf(fr, "%d", &n);
  a = malloc(sizeof(int) * n);
  for (int i =0; i < n; i++){
    fscanf(fr, "%d", &a[i]);
  }

  qsort(a, n, sizeof(int), cmpfunc);

  fw = fopen(output_buf, "w");
  for (int i=0; i<n; i++){
    fprintf(fw, "%d ", a[i]);
    if (i > 0 && i % 100 == 0)
      fprintf(fw, "\n");
  }
  fclose(fw);

}