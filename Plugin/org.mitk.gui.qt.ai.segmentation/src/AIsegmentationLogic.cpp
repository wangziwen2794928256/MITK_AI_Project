#include "AIsegmentationLogic.h"


#include <QProcess>

#include <QDebug>



AIsegmentationLogic::
AIsegmentationLogic(QObject* parent)

:
QObject(parent)

{

}



bool AIsegmentationLogic::
RunSegmentation()

{


return CallPython();


}



bool AIsegmentationLogic::
CallPython()

{


QString python =

"C:/ANACONDA/envs/mitk-ai/python.exe";



QString script =

"D:/MITK_AI_Project/"
"Plugin/org.mitk.gui.qt.ai.segmentation/"
"PythonBridge/run_inference.py";



QStringList args;


args

<< script

<< "input.nii.gz"

<< "output";



QProcess process;



process.start(

python,

args

);



process.waitForFinished(-1);



QString output =

process.readAllStandardOutput();



qDebug()

<< output;



return true;


}