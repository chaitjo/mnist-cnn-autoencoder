#!/bin/bash
source ~/tensorflow_py27_cpu/bin/activate
cd ~/nn-project-2/CNN/
python cnn.py --learning_rate=0.005 --momentum=0.9 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.02 --momentum=0.9 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.05 --momentum=0.9 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.1 --momentum=0.9 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.5 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.75 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.95 --lr_decay=0.95 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.9 --lr_decay=0.9 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.9 --lr_decay=0.75 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.9 --lr_decay=0.5 --batch_size=50
python cnn.py --learning_rate=0.01 --momentum=0.9 --lr_decay=0.95 --batch_size=100
python cnn.py --learning_rate=0.01 --momentum=0.9 --lr_decay=0.95 --batch_size=75
python cnn.py --learning_rate=0.01 --momentum=0.9 --lr_decay=0.95 --batch_size=25