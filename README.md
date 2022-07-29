# lr-range-test
To run lr range test, type in terminal and run
```
CUDA_VISIBLE_DEVICES=0 python3 lrfinder.py --optimizer <optimizer> --batch-size <batchsize> --num-epoch <num-epoch(5)>
```
low and high in function **lrs()** define the search scope
  
  
  
Then get the generated log's filename, run
```
python3 plot_lr_range_test.py --filename <filename>
```
to get the plot
