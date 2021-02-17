function boundaryBench_sintel(flowGTFile, mbFile, evFile)

thinpb = true;
maxDist = 0.0075;
nthresh = 99;


clear gt;
%evFile = fullfile(outDir, strcat(iids(i).name,'_ev1.txt'));
%iids(i).name

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
%bdry2  = readFloat(mbFile);
disp(mbFile);
bdry2 = load(mbFile);
bdrys=bdry2.data;
sz2= 2;
E = single(bdrys);
Es = convTri(E,sz2);

size(Es)
[Ox,Oy]   = gradient2(Es);
[Oxx,~]   = gradient2(Ox);
[Oxy,Oyy] = gradient2(Oy);
O         = mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
ucm2      = edgesNmsMex(E,O,1,5,1.01,4);

%ucm2 =  edgesNmsMex(single(dx),O,1,5,1.01,5); %--> 0.2223
evaluation_bdry_image_sintel(ucm2, gt, evFile, nthresh, maxDist, thinpb);
