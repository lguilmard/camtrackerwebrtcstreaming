# camtrackerwebrtcstreaming
make an autonomous webcam python tracker that stream to a webrtc server

# to install
sudo apt-get install -y python3 python3-pip && pip3 install -r requirements.txt
# alternative with pipenv
pipenv shell
pipenv install


# run AND CUSTOMIZE 
python3 cadreur/auto_crop.py -v
parameters :
• NN_img_rescale : change face detection depending on subject distance
• Hxscale : change vertical croping
• sizeCare : how much closest face is more captured compared to furthest 
• decay : the highest the slowest the field changes and the more it is stable <> the slowest the more the field is unstable and changes quickly 

# run 
• see result : python3 cadreur/auto_crop.py (press ESQ to stop program)

• pipe stream : python3 cadreur/auto_crop.py -pipe | WHAT YOU WANT (Ctrl-c to stop)


# key bindings
Esc or Ctrl-C to quit

