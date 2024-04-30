     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear     
  for i0=1:5000 
      i0
     iname = sprintf('data/%04d_h.png', i0);
    im = imread(iname);
%         im = rgb2gray(im);
im = im2uint8(im);
im=imresize(im,[256 256]);
imwrite(im,['Height/',num2str(i0) ,'.png'])
  end
  
 for i0=1:5000 
      i0
    iname = sprintf('data/%04d_i2.png', i0);
    im = imread(iname);
im1=im(:,:,1);
im2=im(:,:,2);
im3=im(:,:,3);

im1=imresize(im1,[256 256]);
im2=imresize(im2,[256 256]);
im3=imresize(im3,[256 256]);

imwrite(im1,['SegL1/',num2str(i0) ,'.png'])
imwrite(im2,['SegL2/',num2str(i0) ,'.png'])
imwrite(im3,['SegL3/',num2str(i0) ,'.png'])

 end 
  
for i0=1:1250 
      i0
     iname = sprintf('SemanticSegmentationScars/PixelLabelDatastore/Label_%03d.png', i0);
    im = imread(iname);
%         im = rgb2gray(im);
im=imresize(im,[256 256]);
imwrite(im,['SemanticSegmentationScars/PixelLabelDatastore/',num2str(i0) ,'.png'])
  end