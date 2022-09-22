# MUFJ Data Science Champion Ship

## Score
|id|CV|LB|PLB|memo|
|----|----|----|----|----|
|1|0.659|0.641||miss in FL. almost predictions are 1|
|45|0.769|0.783||deberta-large, 3e-5, loss transition is up-down|
|50|0.803|0.815||deberta-v3-base, 2e-5, sc=None|
|51|0.808|0.820||deberta-v3-base, 2e-5, do=0, sc=None|
|65|0.810|0.824||deberta-v3-base, 2e-5, do=0.15, FGM|
|e5|0.817|0.829||65,65p1,98,98p1,104|


## Computation Time
kfold=5 + all
|model|at|ep|bs|time|memory|
|----|----|----|----|----|----|
|deberta-v3-base|None|4|16|2:50|19803|
||AWP|4|16|3:50||
||FGM|4|16|4:50||
|deberta-base|None|4|16|3:20||
|deberta-large|3|8|4:00||
|roberta-base|None|4|16|2:10|13543|
|roberta-large|None|3|8|3:00||
|xlm-roberta-base|None|4|16|2:10|16311|