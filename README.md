# mbeval
Evaluation code for motion boundaries:
**Occlusions, Motion and Depth Boundaries with a Generic Network for Disparity, Optical Flow or Scene Flow**
(E. Ilg and T. Saikia and M. Keuper and T. Brox published at ECCV 2018)
[[paper]](http://lmb.informatik.uni-freiburg.de/Publications/2018/ISKB18)

# Instructions

* Download the BSDS benchmark [code] (https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)
* Download [Piotr's matlab toolbox] (https://pdollar.github.io/toolbox/)
* Set the correct paths in `boundaryBench_sintel.m` (L3-L4)
* To evaluate a single example use `launch.sh flow_gt_0.flo mb_pred_0.mat eval_logs/mb_eval_0.txt`
* After evaluating all examples, aggregate results using `collect_eval_bdry.m ./eval_logs`
