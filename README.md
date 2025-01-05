# SHM_FDD
SHM_FDD is an embedded appliance of Frecuency Domain Decomposition (FDD) for Structural Health Monitoring (SHM) applying Output Only philosophy with an approach on 
Modal Identification, Modal Shapes and Modal, Assurance Criterion (MAC).

The project is part of a SHM system where the analysis step is performed by the these python scripts. Adquisition system is applied by STM32 uController with 12 bits
ADC resolution but mapped up to 16 bits by hardware oversampling technique. The Number of channel used by the application depends on the resolution you want for modal 
identification. 
We recommend you creating an .txt or .csv file which is used for storage all the information created by the adquisition system. The python script assumes that all
information collected are Column Vector Arrays separated by 'tab' as follow: 

Time  channel1  channel2 [...] channelx


We attached an .txt example file as data file which you can use for test the python Script. Also all .txt and for more information about FDD, SHM, Output Only approach and MAC please refer to: 

Rainieri, C., & Fabbrocino, G. (2014). Operational Modal Analysis of Civil Engineering Structures An Introduction and Guide for Applications. 
New York : Springer Science+Business Media
 
All thanks and credits for Rainieri, C., & Fabbrocino, G. (2014) by the
FDD algorithm and .txt example data file. Without the relevant information and comparison among computed data and data stated on the book it wouldn't have been possible.


Last test (05/January/2025):

Python Version 3.11.2

scipy	1.15.0

numpy	1.26.4

pandas	2.2.1

matplotlib	3.10.0


Any further questions please reach me out by email: 

gerardohuerta1705@gmail.com
twitter: 

https://twitter.com/RoblenHuerta

Guide for users: 

Execute FDD Script

-Select Peaks using: "Ctrl + remain pressing left mouse button until draw a rectangle over the peak"

-if you want delete the selection just press right mouse button

-Finish Peack picking by pressing scroll mouse button. 

Once done the peak selection the routine creates 3 new figure plots, and print the frequencies and modal shapes.  
