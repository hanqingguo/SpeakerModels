from pyaudio import paInt16

# Signal processing
SAMPLE_RATE = 192000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 2048
BUCKET_STEP = 1
MAX_SEC = 10

# Model
WEIGHTS_FILE = "data/model/weights.h5"
COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE=(NUM_FFT,None,1)

# IO
# ENROLL_LIST_FILE = "test_model/good_enroll15.csv"
# TEST_LIST_FILE = "test_model/good_test15.csv"
# RESULT_FILE = "res/good_enroll15.csv"

enrollNum = 2
ENROLL_LIST_FILE = "ultraSound/good_enroll"+str(enrollNum)+".csv"
TEST_LIST_FILE = "ultraSound/good_test"+str(enrollNum)+".csv"
RESULT_FILE = "res/ult-good_enroll"+str(enrollNum)+".csv"
