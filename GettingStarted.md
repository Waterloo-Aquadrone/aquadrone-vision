## So you've joined the vision team!

## Things to start with
- Basic understanding of python
- basic understanding of ML - especially keras
    - training url
- basic understanding of python packages such as pandas
    - training url

### First task
- clone the git repo
    - git clone url
- create a new branch for your change
    - git checkout -b yourBranchName
- download a video from the repo of training videos [url] - ideally one not yet in timestamps
- Create a csv in timestamps with the name of the video, and timestamps referencing the object at each part of the video
    - use other csv's in that folder for reference
- run video-frames.py on your video folder
- add, commit, and push your new csv folder
    - git add nameOfCSV (or git add . to add everything)
    - git commit nameOfCSV -m "my message" (or git commit . to commit everything added)
    - git push git push --set-upstream origin yourBranchName
    - go to the provided link, and create a pull request to dev (never to master!)
Congrats! You've made your changes to the repo

### Google Colab
Google colab can be used to train code without your own gpu.

### Supervisely
How we label the bounding boxes for our code