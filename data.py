import gdown
import tarfile

url = 'https://drive.google.com/uc?id=1GsC6vtBm47kNUPrCU-8_hXU1Pi0BraE9'
output = '/home/student/train.tar'
gdown.download(url, output, quiet=False)
tf = tarfile.open(output)
tf.extractall('/home/student/')

url = 'https://drive.google.com/uc?id=1JVCAqZOhKCs3_5KrlZakB-ZJZTcyZSS3'
output = '/home/student/test.tar'
gdown.download(url, output, quiet=False)
tf = tarfile.open(output)
tf.extractall('/home/student/')