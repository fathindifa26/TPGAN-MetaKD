find dataset/cfp-dataset/Data/Images/ -type f -path "*/profile/*.jpg" | sort > cfp_profile_input.txt
find dataset/cfp-dataset/Data/Fiducials/ -type f -path "*/profile/*.txt" | sort > cfp_profile_landmark.txt
find dataset/cfp-dataset/Data/Images/ -type f -path "*/frontal/*.jpg" | sort > cfp_frontal_input.txt

python test.py --frontal_list cfp_frontal_input.txt -input_list cfp_profile_input.txt -landmark_list cfp_profile_landmark.txt -resume_model save/try_68 -subdir cfp_test --batch_size 2