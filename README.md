# camtrackerwebrtcstreaming
make an autonomous webcam python tracker that stream to a webrtc server

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
• see result : python3 path/auto_crop.py (press ESC to stop program)

[ !! not tested !! ] • pipe stream : python3 path/auto_crop.py -pipe | WHAT YOU WANT (Ctrl-c to stop) 

suggestion : python path/auto_crop.py -pipe | cvlc --demux=rawvideo --rawvid-fps=25 --rawvid-width=1280 --rawvid-height=720  --rawvid-chroma=RV24 - --sout "#transcode{vcodec=h264,vb=200,fps=25,width=1280,height=720}:rtp{dst=10.10.10.10,port=8081,sdp=rtsp://10.10.10.10:8081/test.sdp}"


# key bindings
Esc or Ctrl-C to quit

