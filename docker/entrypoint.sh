#!/bin/bash
echo "--- Setting up container, please wait..."

echo "--- Now as root..."
su - root <<!
root
service ssh start
!
echo "--- End of root..."

#echo "--- Additional installations..."
#... space for additional installations

echo "--- Additional container setup completed, ready for work..."
tail -F /dev/null
