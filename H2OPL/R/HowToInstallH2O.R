############################################################
# Dylan's way to debug the reason I can't run h2o
# In the run of h2o "localH2O = h2o.init()", it says
# "Note:  In case of errors look at the following log files:
#/var/folders/z4/_s1ks61s3tx7tz2g3d5zxbmr0000gp/T//RtmpzoDJq7/h2o_uncxu_started_from_r.out
#/var/folders/z4/_s1ks61s3tx7tz2g3d5zxbmr0000gp/T//RtmpzoDJq7/h2o_uncxu_started_from_r.err"
#,
# so dylan look into the error file by doing the following
############################################################

cd /var/folders/z4/_s1ks61s3tx7tz2g3d5zxbmr0000gp/T//Rtmp65fePj/
ls
ls -l
more h2o_uncxu_started_from_r.err 

# The error said it's java's problem, then check java and download jdk java 8, then you go.
#
java -version
more ls -A
more .bash_profile

##=====================================================================================================##
# Two main thing about using h2o r package

# 1. The h2o cloud is in the localhost: ip= 127.0.0.1, port = 54321
# If you don't have the access to it. in Preference -> Network -> Advanced -> Proxy -> Bypass proxy settings for those Hosts & Domains, ass :, 127.*
# 2. Make sure your jave is 1.7 or higher
# type java -version in the termial to check. If it is not 1.7 or higher, download jdk java.

##=====================================================================================================##

# After that, install the package and try

localH2O = h2o.init()