# FuCoLoT: A Fully-Correlational Long-Term Tracker

Matlab implementation of Fully Correlational Long-Term Tracker published at the 14th Asian Conference on Computer Vision (ACCV) 2018.

## Publication
Alan Lukežič, Luka Čehovin Zajc, Tomáš Vojíř, Jiří Matas and Matej Kristan.  ''FuCoLoT - A Fully-Correlational Long-Term Tracker.'' Asian Conference on Computer Vision (ACCV), 2018. </br>
[Paper](http://prints.vicos.si/publications/366) </br>

<b>BibTex citation:</b></br>
@InProceedings{Lukezic_ACCV_2018,<br>
Title = {FuCoLoT - A Fully-Correlational Long-Term Tracker},<br>
Author = {Luke{\v{z}}i{\v{c}}, Alan and {\v{C}}ehovin Zajc, Luka and Voj{\'i}{\v{r}}, Tom{\'a}{\v{s}} and Matas, Ji{\v{r}}{\'i} and Kristan, Matej},<br>
Booktitle = {ACCV},<br>
Year = {2018}<br>
}

## Source code running
* Clone git repository: </br>
    $ git clone https://github.com/alanlukezic/fucolot.git
* Compile mex files running compile.m command in the CSRDCF folder </br>
	Set <i>opencv_include</i> and <i>opencv_libpath</i> to the correct OpenCV paths in the compile.m script
* Use run_on_uav.m script for running the tracker on UAV dataset (https://ivul.kaust.edu.sa/Pages/Dataset-UAV123.aspx) </br>
	Set <i>tracker_path</i> variable to the directory where your source code is and <i>dataset_path</i> to the directory where you have stored the UAV dataset. Set <i>results_path</i> variable to the directory where you want to store the raw results. Modify <i>parfor</i> into <i>for</i> command to disable parallel processing of the sequences.
