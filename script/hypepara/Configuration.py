
class GeneralCon():
    NUM_CLASSES = 4

    IMAGE_SIZE = (224,224)

    BATCH_SIZE = 32

    DATA_FAKING_TRAIN = 30

    DATA_FAKING_VAL = 10

    ATTEMPT = '06'

    FREEZE = True
    
    UNIVERSE = False


class VGGCon(GeneralCon):
    LEARNING_RATE = 0.001

    FREEZE_LAYER = 200

    EPOCH = 10


class ResNetCon(GeneralCon):
    LEARNING_RATE = 0.001

    FREEZE_LAYER = 200

    EPOCH = 10


class MobileNetCon(GeneralCon):
    LEARNING_RATE = 0.001

    FREEZE_LAYER = 200

    EPOCH = 10
    
    
