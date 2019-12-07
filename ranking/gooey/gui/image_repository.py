'''
Collection of the image paths.

The module is meant to act as a singleton, hence the globals() abuse.

Image credit: kidcomic.net
'''
import os
from functools import partial

from ranking.gooey import getResourcePath
from ranking.gooey import merge

filenames = {
    'programIcon': 'program_icon.ico',
    'successIcon': 'company_logo.png',
    'runningIcon': 'company_logo.png',
    'loadingIcon': 'loading_icon.gif',
    'configIcon': 'company_logo.png',
    'errorIcon': 'company_logo.png'
}

def loadImages(targetDir):
    defaultImages = resolvePaths(getResourcePath('images'), filenames)
    return {'images': merge(defaultImages, collectOverrides(targetDir, filenames))}


def getImageDirectory(targetDir):
    return getResourcePath('images') \
           if targetDir == 'default' \
           else targetDir


def collectOverrides(targetDir, filenames):
    if targetDir == '::gooey/default':
        return {}

    pathto = partial(os.path.join, targetDir)
    if not os.path.isdir(targetDir):
        raise IOError('Unable to find the user supplied directory {}'.format(
            targetDir))

    return {varname: pathto(filename)
            for varname, filename in filenames.items()
            if os.path.exists(pathto(filename))}


def resolvePaths(dirname, filenames):
    return {key:  os.path.join(dirname, filename)
            for key, filename in filenames.items()}


