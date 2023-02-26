This is an AutoEncoder-based model for music labeling task. We label music files with tags like instrument type, tonal characteristics, mood characteristics represented by music. The output will be a probability array where each element is a float value between 0-1, indicating the probability of that label.

To train the model and preprocessing the musicfile:
    
    The main file will be train.py and utils.py
   

To evaluating the music:
 
    The main file will be test.py


    '''
    parser.add_argument('--testMusicPath',type=str,default="./dataset/who-bargain.WAV",help='the music we would like to test')
    parser.add_argument('--resultPath', type=str,
                        default="./loggers/eval_result", help='the path we would like to store the eval result')
    '''


    These two are the only arguments you need to mention when in the eval mode
    testMusicPath should be a path of a WAV file
    resultPath should ba an existing folder's path


    Call the pipeline using:
           python test.py --testMusicPath {your path} --resultPath {your path}
    

    Result format:
        
        The file name will be {Month}{Day}_{Hour}{Min}_result.txt
        The file conent will be as follows:

        -----------------------------TESTING MODE------------------------------
        [ 0.48530743  0.07003193  0.0924342   0.11470391  0.30507228 -0.02546413
        0.05761781  0.4103659   0.3782717   0.55372876  0.21798202  0.15904206
        0.4109761   0.1353574   0.23477753  0.28567404  0.1687552   0.10121489
        0.3973233   0.07387893  0.22707799  0.35002837  0.1639494   0.40118986
        0.14184679  0.6004053   0.8250232   0.19624032  0.19369954  0.25266826
        0.15686527  0.6855912   0.35047075  0.42106113  0.15077475  0.01120785
        0.17347932  0.09510875  0.06171694  0.34465954  0.39419654  0.4029168
        0.05216709  0.56101763  0.07052027  0.22321622  0.16953188  0.2083317
        0.4060112   0.17587788  0.2410554   0.8326196   0.10007532  0.1084688
        0.15250549  0.31272528  0.30795816  0.23342001  0.5843073   0.5209879
        0.32068077  0.27401948  0.22180507  0.2538147   0.441807    0.31364694
        0.09135973  0.28979212  0.04662536  0.2471594   0.41196108  0.34157473
        0.06142202  0.09551197  0.6412008   0.13458075  0.34887052  0.1067346
        0.45906386  0.13535821  0.09117573  0.06036995  0.11767799  0.06583814
        0.31095815  0.9040059   0.07021416  0.35964805  0.22306415  0.13191722
        0.18609147  0.45633537  0.22199854  0.6191386   0.31475437  0.07890123
        0.16424255]
        ----------------------------TESTING FINISH-----------------------------
