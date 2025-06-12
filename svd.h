#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include"stdlib.h"
#include"math.h"

void ppp(double a[], double e[], double s[], double v[], int m, int n);
void sss(double fg[2], double cs[2]);
int muav(double a[], int m, int n, double u[], double v[], double eps, int ka1);
int ginv(double a[], int m, int n, double aa[], double eps, double u[], double v[], int ka1);
