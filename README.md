# MDM: ControlPose

this is going to be a quick readme guide to run the project 

# init

after pulling the project follow the instructions from the MDM repository, this can also be found in the file README_MDM in this repository, to run inference only you can download the Text Dataset only 

to train/retrain one of the models you need the full Dataset as instructed in the MDM repository


# running inference

to use the model to produce animations, you first need a pretrained model, for that you can get the one we used for our testing and evaluation from: https://drive.google.com/file/d/1HKywEFxuWs_jnO--N_uBsWI82ALrkm59/view?usp=sharing 

place it at the project directory at /save/your_model_name


running a prediction can be done using 

you can find conditioning images in the sample_conditioning_images folder

python -m sample.generate --model_path path/to/checkpoint.pt --text_prompt "a person raises his right hand" --output_dir "/path/to/output/directory" --cond_path "/path/to/image/condition"

if this gives an error due to a weights mimatch, switch to the presentation branch

