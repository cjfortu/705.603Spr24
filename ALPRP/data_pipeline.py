import numpy as np
import ffmpeg
from collections import deque

class Pipeline:
    """
    A class to extract, transform, and save/load a UDP video stream.
    
    Requires prior knowledge of the:
    UDP stream url
    w/h of the video
    duration
    desired frame slicing regions
    desired frame rate to save
    """
    
    def __init__(self, input_url='udp://127.0.0.1:23000',
                 width=3840, height=2160,
                 trimsize=[[880, 2160], [640, 3200]],
                 viddur=60, dfps=2, npypth='./data/trimarr.npy'):
        self.fullarr = None
        self.input_url = input_url
        self.width = width
        self.height = height
        self.trimsize = trimsize
        self.viddur = viddur
        self.dfps = dfps
        self.npypth = npypth
        self.trimarr = None
        
        
    def extract(self):
        """
        Stream video from a given input URL using ffmpeg and save it as a numpy.npy file.

        source:
        https://github.com/EN-705-603-SP24/Object_Detection_Systems/blob/main/deployment_udp_client.py
        """
        print('Waiting for UDP video stream...')
        process1 = (
            ffmpeg
            .input(self.input_url)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        firstframe = True
        frames = deque()
        while True:  # while the video is still streaming
            in_bytes = process1.stdout.read(self.width * self.height * 3)
            if firstframe == True:
                print('UDP stream detected, reading {}s of video...'.format(self.viddur))
                firstframe = False
            if not in_bytes:
                break
            in_frame = np.frombuffer(in_bytes, np.uint8).\
                    reshape([self.height, self.width, 3])
            frames.append(in_frame)
        fullarr = np.stack(frames, axis = 0)
        print('video of dimension {} received'.format(fullarr.shape))
        process1.wait()
    
        self.fullarr = fullarr
        
        
    def transform(self):
        """
        Slice each frame and reduce the frame rate.
        """
        print('reducing fps and slicing frames...')
        spf = self.viddur / self.fullarr.shape[0]  # seconds per frame
        frames = deque()
        fnum = 0  # initialize frame count
        dfcut = 0  # initialize desired framerate cutoff number
        # define frame slicing regions
        top = self.trimsize[0][0]
        bottom = self.trimsize[0][1]
        left = self.trimsize[1][0]
        right = self.trimsize[1][1]
        for i in range(0, self.fullarr.shape[0]):  # iterate across all possible frames
            if int(i*2*spf) == dfcut:  # fps reduction criteria
                frame = self.fullarr[i, top:bottom, left:right, :]
                frames.append(frame)
                dfcut += 1
            fnum += 1

        self.trimarr = np.stack(frames, axis = 0)
        print('fps reduction and slicing complete...')
        
        
    def load(self):
        """
        load (aka save) the sliced/reduced video as a numpy.npy file.
        """
        np.save(self.npypth, self.trimarr)
        print('transformed video saved, dimensions: {}'.format(self.trimarr.shape))

        
# if __name__ == "__main__":
#     pipeline = Pipeline(input_url='udp://127.0.0.1:23000',
#                  width=3840, height=2160,
#                  trimsize=[[880, 2160], [640, 3200]],
#                  viddur=60, dfps=2, npypth='./data/trimarr.npy')
#     pipeline.extract()
#     pipeline.transform()
#     pipeline.load()