python make.py --mode fs --run train --num_experiments 4 --round 12
python make.py --mode fs --run test --num_experiments 4 --round 12

python make.py --mode ps --run train --num_experiments 4 --round 12
python make.py --mode ps --run test --num_experiments 4 --round 12

python make.py --mode fl --run train --num_experiments 4 --round 12 --split_round 2
python make.py --mode fl --run test --num_experiments 4 --round 12 --split_round 2

python make.py --mode ssfl --run train --num_experiments 4 --round 12 --split_round 2
python make.py --mode ssfl --run test --num_experiments 4 --round 12 --split_round 2

python make.py --mode frgd --run train --num_experiments 4 --round 12 --split_round 2
python make.py --mode frgd --run test --num_experiments 4 --round 12 --split_round 2

python make.py --mode fmatch --run train --num_experiments 4 --round 12 --split_round 2
python make.py --mode fmatch --run test --num_experiments 4 --round 12 --split_round 2

python make.py --mode tau --run train --num_experiments 4 --round 12
python make.py --mode tau --run test --num_experiments 4 --round 12

python make.py --mode mix --run train --num_experiments 4 --round 12
python make.py --mode mix --run test --num_experiments 4 --round 12

python make.py --mode lu --run train --num_experiments 4 --round 8 --split_round 2
python make.py --mode lu --run test --num_experiments 4 --round 8 --split_round 2

python make.py --mode lu-s --run train --num_experiments 4 --round 8 --split_round 2
python make.py --mode lu-s --run test --num_experiments 4 --round 8 --split_round 2

python make.py --mode gm --run train --num_experiments 4 --round 8
python make.py --mode gm --run test --num_experiments 4 --round 8

python make.py --mode sbn --run train --num_experiments 4 --round 8
python make.py --mode sbn --run test --num_experiments 4 --round 8

python make.py --mode alternate --run train --num_experiments 4 --round 20 --split_round 2
python make.py --mode alternate --run test --num_experiments 4 --round 20 --split_round 2