#include "AIsegmentationView.h"


#include "ui_AIsegmentationView.h"


#include <QMessageBox>



const std::string AIsegmentationView::VIEW_ID =
"org.mitk.gui.qt.ai.segmentation";



void AIsegmentationView::CreateQtPartControl(
        QWidget* parent)

{


ui =
new Ui::AIsegmentationView;


ui->setupUi(parent);



m_Logic =
new AIsegmentationLogic();



connect(

ui->startButton,

SIGNAL(clicked()),

this,

SLOT(OnStartSegmentation())

);



}



void AIsegmentationView::SetFocus()
{


}



void AIsegmentationView::OnStartSegmentation()

{


ui->progressBar->setValue(10);



bool result =
m_Logic->RunSegmentation();



if(result)

{

ui->progressBar->setValue(100);


QMessageBox::information(

nullptr,

"AI",

"Segmentation Finished"

);


}


else

{


QMessageBox::warning(

nullptr,

"AI",

"Failed"

);


}



}