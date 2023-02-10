# accept hdr filename
convert_one () {
	fslchfiletype NIFTI $1 &
	sleep 0.1
	echo "$1 has been converted to nii format"
}

	
hdrs=$(find . -name "s*hdr")
for file in $hdrs
do
	convert_one $file
done

