# camtrackerwebrtcstreaming
•1st make an autonomous webcam cameraman like algorithm ( adjust the field of view to see every persons in scene)
•[TODO] 2nd use output in streamer

# to install
    sudo apt-get install -y python3 python3-pip && pip3 install -r requirements.txt
# alternative with pipenv
    pipenv shell
    pipenv install


# run AND CUSTOMIZE 
    python3 path/auto_crop.py -v

    GUI parameters :
        
    • NN_img_rescale : change face detection depending on subject distance
    • Hxscale : change vertical croping
    • sizeCare : how much closest face is more captured compared to furthest 
    • decay : the highest the slowest the field changes and the more it is stable <> the slowest the more the field is unstable and changes quickly 

# run 
• see result : 

    python3 path/auto_crop.py (press ESC to stop program)

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


