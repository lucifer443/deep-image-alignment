python tools/test.py $1 $2 --out work_dirs/tmp_h.pkl 

python tools/evaluate.py work_dirs/tmp_h.pkl $3 

rm work_dirs/tmp_h.pkl
