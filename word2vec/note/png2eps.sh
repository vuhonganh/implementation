#!/bin/sh

# File: png2eps.sh: convert every .png file to .eps if the latter does not exist

for FILE in `find ./images -name '*.png'`
do
    FILENAME=`basename $FILE .png`;
    if [ ! -e ./images/${FILENAME}.eps ]; then
	echo  "Converting  $FILENAME.png to  $FILENAME.eps"
	convert ./images/${FILENAME}.png eps3:./images/${FILENAME}.eps
	if [ $? -eq 0 ]; then
	    echo "Converting DONE"
	else
	    echo "Converting FAILED. Make sure convert program is available (install ImageMagick by sudo yum install ImageMagick)."
	fi
    fi
    
done

