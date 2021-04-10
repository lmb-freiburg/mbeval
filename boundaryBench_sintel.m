function boundaryBench_sintel(flowGTFile, mbFile, evFile)

addpath(genpath('/pathto/pdollar_toolbox/toolbox'));
addpath(genpath('/pathto/BSR/bench/benchmarks'));

thinpb = true;
maxDist = 0.0075;
nthresh = 99;

clear r;
clear s;
clear gt;

flo=readFlowFile(flowGTFile);

[dx1,dy1] = gradient(flo(:,:,1));
[dx2,dy2] = gradient(flo(:,:,2));
dx=sqrt(dx1.*dx1+dy1.*dy1 + dx2.*dx2+dy2.*dy2);
thrs=[1,2,4,8];


for l=1:length(thrs)
    if(thrs(l)<=max(min(thrs),max(dx(:))))
        gt{l} = dx>=thrs(l);
        gt{l}=double(bwmorph(gt{l}  , 'thin', inf));
    end
end

% read mb
disp(mbFile);
bdry2 = load(mbFile);
bdrys=bdry2.data;
sz2= 2;

%figure,imshow(bdrys,[])
E = single(bdrys);
max(E(:))

%display(size(E));
Es = convTri(E,sz2);
size(Es)
[Ox,Oy]   = gradient2(Es);
[Oxx,~]   = gradient2(Ox);
[Oxy,Oyy] = gradient2(Oy);
O         = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
ucm2      = edgesNmsMex(E,O,1,5,1.01,4);
evaluation_bdry_image_sintel(ucm2, gt, evFile, nthresh, maxDist, thinpb);
