function boundaryBench_sintel(imgDir, gtDir, pbDir, outDir, i,thresh, maxDist, thinpb)
% boundaryBench(imgDir, gtDir, pbDir, outDir, nthresh, maxDist, thinpb)
%
% Run boundary benchmark (precision/recall curve) on dataset.
%
% INPUT
%   imgDir: folder containing original images
%   gtDir:  folder containing ground truth data.
%   pbDir:  folder containing boundary detection results for all the images in imgDir. 
%           Format can be one of the following:
%             - a soft or hard boundary map in PNG format.
%             - a collection of segmentations in a cell 'segs' stored in a mat file
%             - an ultrametric contour map in 'doubleSize' format, 'ucm2' stored in a mat file with values in [0 1].
%   outDir: folder where evaluation results will be stored
%	nthresh	: Number of points in precision/recall curve.
%   MaxDist : For computing Precision / Recall.
%   thinpb  : option to apply morphological thinning on segmentation
%             boundaries before benchmarking.
%
% based on boundaryBench by David Martin and Charless Fowlkes:
% http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/code/Benchmark/boundaryBench.m
%
% Pablo Arbelaez <arbelaez@eecs.berkeley.edu>
% addpath VSB/bvds/
% addpath VSB
% addpath /BS/metric-cut/work/BSR/bench/benchmarks/
% addpath ~/software/motionSeg/flow-code-matlab/
% 
% addpath ~/software/structuredRF/toolbox/channels/
% addpath ~/software/structuredRF/toolbox/filters/ 
% addpath ~/software/structuredRF/toolbox/        
% %channels/  classify/  detector/  doc/       external/  filters/   images/    matlab/    videos/    
% addpath ~/software/structuredRF/toolbox/external/
% addpath ~/software/structuredRF/toolbox/images/  
% addpath ~/software/structuredRF/  

if nargin<8, thinpb = true; end
if nargin<7, maxDist = 0.0075; end
if nargin<6, nthresh = 99; end

%seqlist=dir(gtDir);
%seqlist=seqlist(3:end);
%for i=1:numel(seqlist)
%flowlist{i}=dir([gtDir,'/',seqlist(i).name,'/','*.flo']);
%end
mkdir(outDir)
%iids = dir(fullfile(imgDir,'*img1.ppm'))
iids = dir(fullfile(imgDir,'*.flo'))
numel(iids)

%for i = 1:numel(iids),
%k=1;
%for i = 1:numel(seqlist)
    
%    for j=1:numel(flowlist{i})
        clear gt;
    %load(['/BS/MRI-segmentation/work/MOSEG/Test', '/',seqlist(i).name,'/imageucms.mat'])
    evFile = fullfile(outDir, strcat(iids(i).name,'_ev1.txt'));
    %iids(i).name(1:end-4),'_ev1.txt'));
    %if exist('evFile','file'), continue; end
iids(i).name
    %[imgDir,'/',iids(i).name(1:end-8),'mb_fwd_gt.float3']
%    [gtDir,'/',seqlist(i).name,'/',flowlist{i}(j).name]
    %flo=readFlowFile([gtDir,'/',iids(i).name(1:end-8),'flow0-gt.flo']);
    flo=readFlowFile([gtDir,'/',iids(i).name(1:end-3),'flo']);
    %figure,imshow(flowToColor(flo))
    [dx1,dy1] = gradient(flo(:,:,1));
    [dx2,dy2] = gradient(flo(:,:,2));
    dx=sqrt(dx1.*dx1+dy1.*dy1 + dx2.*dx2+dy2.*dy2);
    thrs=[1,2,4,8]
    dxmax=max(dx(:));
    for l=1:length(thrs)
        if(thrs(l)<=max(min(thrs),max(dx(:))))
      
     gt{l} = dx>=thrs(l);
     gt{l}=double(bwmorph(gt{l}  , 'thin', inf));
        end
    end
    
    
    %gt=readFloat3([imgDir,'/',iids(i).name(1:end-8),'mb_fwd_gt.float3']);
    %bdry2  = readFloat3([imgDir,'/',iids(k).name(1:end-8),'pred_mb_soft_bwd.float3']);
%try    
%bdry2  = double(imread([imgDir,'\',iids(i).name(1:end-3),'png']))/255;
bdry2  = readFlowFile([imgDir,'/',iids(i).name(1:end-3),'flo']);
%end
%figure,imshow(bdry2,[])
 %try
     %[imgDir,'/',iids(k-1).name(1:end-8),'pred_mb_soft_bwd.float3']
 %   bdry3  = readFloat3([imgDir,'/',iids(i-1).name(1:end-8),'pred_mb_soft_bwd.float3']);
    %figure,imshow(bdry3,[])
 %   bdry2=(bdry2+bdry3)/2;
%    end
    bdrys=bdry2;%(:,1:end-3);
    %bdrys=[zeros(size(bdrys,1),3),(bdrys)];
%bdrys=bdry3;

    %bdry  = imresize(imread([imgDir,'/',seqlist(i).name,'/',gtFilename(1).name(1:end-16),'-fnet2clean_mb_soft_fwd.float3.bmp']),size(groundTruth{1}.Boundaries));
    %ucm=ucms(3:2:end,3:2:end,floor(size(ucms,3)/2));
    %bdry=imresize(double(bdry),size(ucm));
    %ucm2=255*(1.5*sqrt((bdry/255).*ucm));
%sum(isnan(bdrys(:)))
%if(j==1)
%figure, imshow(bdrys+gt{1},[])
%end
%% old - Margret
sz2= 2;
     E = single(bdrys);
     Es = convTri(E,sz2);
     size(Es)
     [Ox,Oy] =gradient2(Es);
     [Oxx,~] =gradient2(Ox); 
     [Oxy,Oyy]   =gradient2(Oy);
     O           =mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
    ucm2       =edgesNmsMex(E,O,1,5,1.01,4);
 %% new - Taha changed to fit the optical flow with two channels
 
%    [dx1,dy1] = gradient(bdrys(:,:,1));
%    [dx2,dy2] = gradient(bdrys(:,:,2));
%    dx = (sqrt(dx1.*dx1+dy1.*dy1 + dx2.*dx2+dy2.*dy2));
    %dx = dx - 0.02;
    %dx(dx<0.02)=0;
    %dx(dx<0.04)=0;
%    dx=normalize_output(dx);
    %dx=dx.^(1/2);    % 
    %dx=dx/max(dx(:));
    %dx(dx<0.005)=0;
    %dx=dx/8.0;
    %fi=fspecial('gaussian',6,2)
    %dxs=imfilter(dx,fi);
    %dx(dx<0.005)=0;
    
 %   [Ox,Oy]=gradient2(convTri(single(dx),4));
 %   [Oxx,~]=gradient2(Ox); 
 %   [Oxy,Oyy]=gradient2(Oy);
 %   O = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
    %O2=atan2(dy1,dx1)/2 + atan2(dy2,dx2)/2;
    %O2=single(mod(O2+pi,pi));
    % defualt values r = 1, s = 5,m = 1.01
    % numbers used in my paper ucm2 =  edgesNmsMex(single(dx),O,6,5,1.01,4);
    % ucm2 =  edgesNmsMex(single(dx),O,1,3,1.01,1); --> Sleeping_1 = 0.2134
    % ucm2 =  edgesNmsMex(single(dx),O,1,5,1.01,1); --> Sleeping_1 = 0.2134
    % ucm2 =  edgesNmsMex(single(dx),O,1,5,1.01,5); --> Sleeping_1 = 0.2226
    % ucm2 =  edgesNmsMex(single(dx),O,1,5,1.01,15);--> Sleeping_1 = 0.2223
    % ucm2 =  edgesNmsMex(single(dx),O,1,5,1.01,10); --> 0.2223
     ucm2 =  edgesNmsMex(single(dx),O,1,5,1.01,5); %--> 0.2223
   %% 
  % ucm2 = single(double(imread([pbDir,'\',iids(i).name(1:end-3),'png']))/255);
   %y(:,:,:)=255-x(:,:,:);
   % Uncomment to produce motion boundary images directly 
   %imwrite(1 -ucm2(:,:), strcat(outDir,'\',iids(i).name,'_sample.png'));
   %%
   
    %figure, imshow(ucm2+gt,[])
%figure,   imshow(ucm2,[])
%figure,   imshow((tanh(-1)+tanh(2*(ucm2/255-0.5))*255),[])
%ucm2=(1./(1+exp((0.5-double(ucm2))/180)));
%ucm2=(tanh(-1)+tanh(2*(ucm2/255-0.5)));

%ucm2=bdrys;
%load(gtFile)
%figure, imshow(groundTruth{i}.Boundaries)

%im=ucm2;
%im(:,:,2)=imresize(ucm*255,size(ucm2));
%im(:,:,3)=imresize(ucm*255,size(ucm2));
%size(ucm)
%size(ucm2)
%figure,   imshow(im);

    %E_nmx1       =edgesNmsMex(E,O,1,5,1.01,4);
%E_nmx2       =edgesNmsMex(E,O,1,5,1.05,4);
%E_nmx3       =edgesNmsMex(E,O,1,5,1.1,4);
%E_nmx4       =edgesNmsMex(E,O,1,5,1.5,4);
%E_nmx5       =edgesNmsMex(E,O,1,5,2,4);
%E_nmx =0.2* (E_nmx1 +E_nmx2+E_nmx3+E_nmx4+E_nmx5);
    %save([imgDir,'/',seqlist(i).name,'/',gtFilename(1).name(1:end-16),'-fnet2clean_mb_soft_fwd.float3.mat'],'ucm2');
    %imwrite(ucm2,[imgDir,'/',seqlist(i).name,'/',gtFilename(1).name(1:end-16),'-fnet2clean_mb_soft_fwd.float3_nmx.bmp']);
  %  fullfile(imgDir,'/',seqlist(i).name,gtFilename(1).name(1:end-16),'-fnet2clean_mb_soft_fwd.float3.mat')
  %  inFile = [imgDir,'/',seqlist(i).name,'/',gtFilename(1).name(1:end-16),'-fnet2clean_mb_soft_fwd.float3.mat'];
    %if ~exist(inFile,'file'),
    %    inFile = fullfile(pbDir, strcat(iids(i).name(1:end-4),'.png'));
    %end
    %gtFile = fullfile(gtDir, strcat(iids(i).name(1:end-4),'.mat'));
    
    
    
    
    % Uncomment to evaluate I have to do this to produce evaluation txt
    % files
    evaluation_bdry_image_sintel(ucm2,gt, evFile, nthresh, maxDist, thinpb);
    disp([i]);
 %   k=k+1;
%    end
   
%end

%% collect results
%collect_eval_bdry(outDir);

%% clean up
%system(sprintf('rm -f %s/*_ev1.txt',outDir));



function [pb_norm] = normalize_output(pb)
% map ucm values to [0 1] with sigmoid 
% learned on BSDS
[tx, ty] = size(pb);
beta = [-1.7487; 3*11.1189];
pb_norm = pb(:);
x = [ones(size(pb_norm)) pb_norm]';
pb_norm = 1 ./ (1 + (exp(-x'*beta)));
pb_norm = (pb_norm - 0.0602) / (1 - 0.0602);
pb_norm=min(1,max(0,pb_norm));
pb_norm = reshape(pb_norm, [tx ty]);





