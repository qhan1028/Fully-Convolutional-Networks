input=$1
output="${1%.*}_fcn.mov"

# Convert video to images
mkdir .tmp
ffmpeg -i $input -ss 00:00:00 -to 00:00:20 .tmp/im_%5d.png 
echo .tmp > testlist.txt
ls .tmp >> testlist.txt

# Run matting program
CUDA_VISIBLE_DEVICES=1 python3.5 FCN.py -m test -ld logs -rd .tmp --video

# Convert back to video
ffmpeg -y -r 30 -i .tmp/pred_%05d.png -r 30 $output

# Clear tmp files
rm -rf .tmp
rm testlist.txt
