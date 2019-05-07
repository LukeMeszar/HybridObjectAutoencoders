#!/bin/bash
TMPFILE=`mktemp`
PWD=`pwd`
PWD=$PWD"/data"
wget "https://grantbaker.keybase.pub/data/background_filtered/test_imagefolder.zip" -O $TMPFILE
unzip -d $PWD $TMPFILE
rm $TMPFILE
wget "https://grantbaker.keybase.pub/data/background_filtered/train_imagefolder.zip" -O $TMPFILE
unzip -d $PWD $TMPFILE
rm $TMPFILE

