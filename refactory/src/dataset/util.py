# common library --------------------------------
from src.utils.logger import log
from src.utils.include import *

# import other ----------------------------------



def rescale_image(image, slope, intercept):
    return image * slope + intercept