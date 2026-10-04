#ifndef AISEGMENTATIONVIEW_H
#define AISEGMENTATIONVIEW_H


#include <QmitkAbstractView.h>


#include "AIsegmentationLogic.h"



namespace Ui
{
class AIsegmentationView;
}



class AIsegmentationView :

        public QmitkAbstractView

{


Q_OBJECT


public:


static const std::string VIEW_ID;


protected:


void CreateQtPartControl(QWidget* parent) override;


void SetFocus() override;



private slots:


void OnStartSegmentation();



private:


Ui::AIsegmentationView* ui;


AIsegmentationLogic* m_Logic;


};



#endif