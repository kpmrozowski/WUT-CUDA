
#include "stdio.h" 
#include "math.h" 
#include "stdlib.h" 

int main(int argc, char *argv[]) 
{ 
    short   x, y, count; 
    long double zr, zi, betar, betai, a, b; 
    long double rsquared, isquared, mr, mi, msquared; 

    double LEFT = atof(argv[1]); 
    double BOTTOM = atof(argv[2]); 
    double RIGHT = atof(argv[3]); 
    double TOP = atof(argv[4]); 
    double  SIZE = atof( argv[5] ); 
    double  MaxIters = atof( argv[6] ); 
     
    for (y = 0; y < SIZE; y++)    
    { 
        for (x = 0; x < SIZE; x++)     
        { 
            zr = 1.0; 
            zi = 0.0; 
            betar = LEFT + x * (RIGHT - LEFT) / SIZE; 
            betai = BOTTOM + y * (TOP - BOTTOM) / SIZE; 

            a = 0.5 * (1 - betar);    
            b = -0.5 * betai;       
            rsquared = zr * zr; 
            isquared = zi * zi; 

            for (count = 0; 
              rsquared + isquared >= 2.25/(betar*betar + 
              betai*betai) && count < MaxIters; count++)    
            { 
                mr = 2*a - 1 + exp(a * zr - b * zi) 
                     * cos(a * zi + b * zr); 
                mi = 2*b + exp(a * zr - b * zi) 
                     * sin(a * zi + b * zr); 
                msquared = mr * mr + mi * mi; 
                zr = 1 - 2 * (a * mr + b * mi)/msquared;   
                zi = 2 * (a * mi - b * mr)/msquared;       
                rsquared = zr * zr; 
                isquared = zi * zi; 
            } 
            
            if (rsquared + isquared >=
                              2.25/(betar*betar+betai*betai)) 
                printf("*"); 
            else 
                printf("."); 
        } 
        printf("\n"); 
    } 
    return 0;

 

}