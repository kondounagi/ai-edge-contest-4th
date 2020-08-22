COMPETITION_ID=285
FILE_ID_ARR=(1073 1054 1055 1066 1057)
for file_id in "${FILE_ID_ARR[@]}" do
  signate download --competition-id="${COMPETITION_ID}" --file-id="${file_id}"
done
