#!/bin/bash

cd /home/saikiat/myrepo/mbeval

/misc/software/matlab/R2019a/bin/matlab -batch "boundaryBench_sintel('$1', '$2', '$3')"
