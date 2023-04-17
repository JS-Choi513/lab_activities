#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
 
#ifndef __linux__
#include <windows.h>
#define fopen64 fopen
#endif
 
#define SEC_PER_PAGE 8
#define MAX_GIGA 1024
int main(int argc,char *argv[]){
    FILE *fp, *fp2, *fp3;
    unsigned int sec, nano = 0, sector = 0, num = 0, page = 0, page_num;
    char rw[3], action = 'D';
    unsigned long long misaligned = 0, lines = 0, range, misrange = 0;
    unsigned long long write_count = 0, read_count = 0;
    char line[1024];
    char str[1024];
    int scaled_count = 0;
    int len;
    int remain = 0;
 
    if(argc < 3)
        return 0;
 
    //printf(" %s \n", argv[1]);
 
    fp = fopen64(argv[1], "r");
    if(fp == NULL){
        printf(" Cannot open file %s\n", argv[1]);
        return 0;
    }
 
    fp2 = fopen64(argv[2], "w");
    if(fp2 == NULL){
        printf(" Cannot open file %s\n", argv[2]);
        return 0;
    }
 
    sprintf(str, "scaled_%s", argv[2]);
 
    fp3 = fopen64(str, "w");
    if(fp3 == NULL){
        printf(" Cannot open file %s\n", str);
        return 0;
    }
 
    while(!feof(fp)){
 
        if(fgets(line, 1023, fp)==NULL){
            if (feof(fp)){
                if (strlen(line)==0)
                    return;
            }
            else
                printf("trace format or read error");
        }
        if (feof(fp))
            break;
 
        if(0 < sscanf(line, "%u %u %s %u %u %s %c\n", &sec, &nano, rw, &sector, &num, rw, &action)){
 
            sector -= 7;
 
            //alignment for 4KB block
            page = sector / SEC_PER_PAGE;
            remain = sector % SEC_PER_PAGE;
            num += remain;
            page_num = num / SEC_PER_PAGE;
            if(num % SEC_PER_PAGE){
                page_num++;
                misaligned++;
            }
 
            fprintf(stdout, "%.6f 0 %u %u %u\n", (float)sec*1000+(float)((float)nano/1000)/(float)1000, page * SEC_PER_PAGE, page_num * SEC_PER_PAGE, (rw[0] == 'W') ? 0 : 1);
 
            /* Check Parameters */
            if(sector == 0)
                continue;
            if(page_num == 0)
                continue;
            if(page_num > 4096)
                continue;
 
            range = (unsigned long long)sector * (unsigned long long)512;
 
            if((unsigned long long)range >= (unsigned long long)MAX_GIGA*1024*1024*1024){
                misrange++;
                printf("Misrange %fGB\n", (double)range/(double)(1024*1024*1024));
                fprintf(stdout, "%.6f 0 %u %u %u\n", (float)sec*1000+(float)((float)nano/1000)/(float)1000, page * SEC_PER_PAGE, page_num * SEC_PER_PAGE, (rw[0] == 'W') ? 0 : 1);
                fprintf(stdout, "remain = %d \n", remain);
                continue;
            }
 
            /* Print converted format */
            
            fprintf(fp2, "%.6f 0 %u %u %u\n", (float)sec*1000+(float)((float)nano/1000)/(float)1000, page * SEC_PER_PAGE, page_num * SEC_PER_PAGE, (rw[0] == 'W') ? 0 : 1);
 
            if(lines%100 == 0){
                fprintf(fp3, "%.6f  %d\n", (float)sec*1000+(float)((float)nano/1000)/(float)1000, page * SEC_PER_PAGE);
                scaled_count++;
            }
 
            /* Sample trace */
            if((unsigned long long)lines < (unsigned long long)10){
                fprintf(stdout, "%.6f 0 %u %u %u\n", (float)sec*1000+(float)((float)nano/1000)/(float)1000, page * SEC_PER_PAGE, page_num * SEC_PER_PAGE, (rw[0] == 'W') ? 0 : 1);
                fprintf(stdout, "remain = %d \n", remain);
 
            }
 
            if(rw == 'W')
                write_count++;
            else
                read_count++;
 
            lines++;
        }
 
    }
 
    fclose(fp);
    fclose(fp2);
    fclose(fp3);
 
    printf(" Write count = %llu\n", write_count);
    printf(" Read count = %llu\n", read_count);
    printf(" Misaligned page = %llu\n", misaligned);
    printf(" Misrange = %llu\n", misrange);
    printf(" Total lines = %llu\n", lines);
 
    return 0;
}