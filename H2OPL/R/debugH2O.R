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