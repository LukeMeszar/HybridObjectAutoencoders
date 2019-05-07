#!/bin/bash
if [ $1 -eq 2 ]
then
  TMPFILE=`mktemp`
  PWD=`pwd`
  wget "https://grantbaker.keybase.pub/data/background_filtered/test_imagefolder.zip" -O $TMPFILE
  unzip -d $PWD $TMPFILE
  rm $TMPFILE
  wget "https://grantbaker.keybase.pub/data/background_filtered/train_imagefolder.zip" -O $TMPFILE
  unzip -d $PWD $TMPFILE
  rm $TMPFILE
fi
python CNN_VAE.py $1
