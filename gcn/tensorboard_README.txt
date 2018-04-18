the whole model based system--easily save and load models
so this means
call your code using

python train.py --output_name my_new_model_name --input_name my_restored_model_name

the output_name flag should always be included
the input name flag is optional
the output name flag defines the name of the model you are saving
the input name flag defines the model you are initializing your parameters from (to train on top of a previous training)

when your training finishes
it saves a model in models/MODEL_NAME/.
and also saves a tensoboard file in tensorboard/MODEL_NAME/.
just to remind you, you can open tensorboard with:

tensorboard --logdir PATH_TO_GCN/TENSORBOARD/.
when you go to the ip it gives you

you'll see something like this [img]
notice in the bottom left you can tick and untick particular models to compare your parameters
those are the same names as what you specify in the command line argument

to add a variable you want to track, modify the build_summaries function in line 48 of train.py (just copy the format I've given there),and then modify line 156 to actually output the new quantity

 it is set to generate the tensorboard data only at the end of the code
however if you are running a long piece of code, you will want it to update live as you go
in which case you need to take the flush statement in line 206 and move it to right after "summary_writer.add_summary"

but probably make the flush only execute every N epochs because it is a costly operation

make sure you are keeping your model organized by putting a new name EVERY time you run the code
an ideally you are recording what each model name corresponds to