# camtrackerwebrtcstreaming
•1st make an autonomous webcam cameraman like algorithm ( adjust the field of view to see every persons in scene)
•[TODO] 2nd use output in streamer
•[TODO] enhance face detection (make it morelieable with face 3/4 or side view) 

    # development ideas under linux
    # to make auto_crop output appear as webcam
    http://wonderingcode.blogspot.com/2017/05/loopback-with-python-and-opencv.html
    

• MACOSX solution [very unstable :( ]

    # make virtual camera using webcamoid.app (see https://webcamoid.github.io/)
    make tmpfs_DIR be a RAM disk (see https://osxdaily.com/2007/03/23/create-a-ram-disk-in-mac-os-x/)
        Example : diskutil erasevolume HFS+ 'tmpfs_DIR' `hdiutil attach -nomount ram://102400` && ln -s /Volumes/tmpfs_DIR/ ./
    make virtual camera stream tmpfs_DIR/auto_crop_output.png 

# to install
    sudo apt-get install -y python3 python3-pip && pip3 install -r requirements.txt
# alternative with pipenv
    pipenv shell
    pipenv install


# run SEE RESULT AND CUSTOMIZE 
    python3 path/auto_crop.py -v

    GUI parameters :
        
    • NN_img_rescale : change face detection depending on subject distance
    • Hxscale : change vertical croping
    • sizeCare : how much closest face is more captured compared to furthest 
    • decay : the highest the slowest the field changes and the more it is stable <> the slowest the more the field is unstable and changes quickly 
    
![alt text](https://github.com/lguilmard/camtrackerwebrtcstreaming/blob/master/preview.png)

# run threaded fast output on tmpsf 
# (PRESS ENTER TO STOP IT PROPERLY)

    # 3 threads programs 
    • thread 1 <> detect field of view in the webcam image (SLOW)
    • thread 2 <> crops / resize and write image in tmpfs_DIR directory (FAST)
    • thread 3 <> WAIT FOR ENTER KEY TO BE PRESSED (to kill threads properly and release webcam capture)
    python3 path/auto_crop.py -thread (press ENTER in command prompt to stop program properly)
# run and see result

    python3 path/auto_crop.py -thread (press ESC in command prompt to stop program properly)

border of camera field of view is materialized by RED rectangle so that user keep aware of exiting camera field

[ !! not tested !! ] • pipe stream : 

    python3 path/auto_crop.py -pipe | WHAT YOU WANT 

(Ctrl-c to stop) 

    #suggestion : 
    python path/auto_crop.py -pipe | cvlc --demux=rawvideo --rawvid-fps=25 --rawvid-width=1280 --rawvid-height=720  --rawvid-chroma=RV24 - --sout "#transcode{vcodec=h264,vb=200,fps=25,width=1280,height=720}:rtp{dst=10.10.10.10,port=8081,sdp=rtsp://10.10.10.10:8081/test.sdp}"

[ !! not tested !! ] • pipe stream with jpg compression : 

    python3 path/auto_crop.py -pipe-JPG | WHAT YOU WANT 

(Ctrl-c to stop) 

    #suggestion : 
    python path/auto_crop.py -pipe | cvlc --demux=rawvideo --rawvid-fps=25 --rawvid-width=1280 --rawvid-height=720  --rawvid-chroma=RV24 - --sout "#transcode{vcodec=h264,vb=200,fps=25,width=1280,height=720}:rtp{dst=10.10.10.10,port=8081,sdp=rtsp://10.10.10.10:8081/test.sdp}"


# key bindings
Esc or Ctrl-C to quit

# USER HELP INSTALL

under linux open bash terminal and go in path where you want to get the code then do as folow

    sudo apt-get update
    sudo apt-get -y upgrade
    sudo apt install -y git-all python3 python3-pip
    git clone https://github.com/lguilmard/camtrackerwebrtcstreaming.git
    pipenv shell pipenv install
    python3 auto_crop.py -v



