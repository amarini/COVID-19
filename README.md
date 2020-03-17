# COVID-19
analysis scripts and macros for COVID-19

### analyze_italy.py 

it is a small script that analyze json data from  'Protezione Civile' (https://github.com/pcm-dpc/COVID-19)

### simulate.py 

it is a small script that simulate a word as a circular connected places, with probability of infection and recover.
This model is similar to a discrete and probabilistic SIR in each place (with some deviations due to binomial factors when the number of infected people is large). 
The recovery should be seen as the R in SIR models (i.e. when people stop to be contagious, aka hospitalizations, full recovery, or death.
There is also a probability  to move around the places (long and short distance movements).
There is an additional probability of moving back from R->S in case of mutation or non effective immunization. 



