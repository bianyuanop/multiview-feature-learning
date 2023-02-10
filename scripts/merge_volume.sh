folders=$(find . -type d | grep -P "./Sub.*/run.*")

FAIL=0

for folder in $folders
do
	swrafs=($(find $folder -name "s*nii.gz" | sort))
	if [ ${#swrafs[@]} -gt 0 ]; then
		echo "merging to $folder/merged.nii.gz"
		fslmerge -t "$folder/merged.nii.gz" "${swrafs[@]}" &
	fi
done

for job in `jobs -p`
do
	echo "waiting job $job"
	wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
	echo "all done"
else
	echo "failed ($FAIL)"
fi

