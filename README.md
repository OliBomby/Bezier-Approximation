### Bezier-Approximation
Library for approximating arbitrary paths using a single Bezier curve.

## How to use this
- Download all the files in this repository and unpack them in a folder.
- Install Python 3.5.2 and all the requirements.
- Put the .osu code of a slider in input.txt
- Run slider_approximator.py
- Enter the parameters you want or leave them empty for a default value.
- Wait for the program to complete.
- Find your new slider .osu code in output.txt

# Requirements
These are the versions on my computer at the time. Later versions may work too.
- Python 3.5.2
- numpy 1.14.5
- tensorflow 1.10.0
- matplotlib 2.1.0
- scipy 1.0.0

# Parameters
- Number of anchors: The number of anchors to use for the generated bezier curve. (default 500)
- Number of training steps: The number of steps in the path approximation. A higher number of steps generally results in a better approximation but also takes longer. (default 10000)
- Evaluating resolution: The number of points on the bezier curve to evaluate how close they are to the wanted path. This should be high enough to capture every detail of the wanted path. (default 5000)
- Retarded: Magnitude of the random noise to add to the starting positions of the bezier anchors. More Retarded results in more chaotic looking bezier anchors. (default 0)

# Other uses
If you know a little bit of Python you can feed the shapes from shapes.py into the shape approximator to generate cool sliders with mathematical shapes. Have fun!
