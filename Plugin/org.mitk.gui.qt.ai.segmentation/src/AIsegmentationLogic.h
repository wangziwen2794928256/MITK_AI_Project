#ifndef AISEGMENTATIONLOGIC_H
#define AISEGMENTATIONLOGIC_H


#include <QObject>


class AIsegmentationLogic :

        public QObject

{


Q_OBJECT



public:


explicit AIsegmentationLogic(QObject* parent=nullptr);



bool RunSegmentation();



private:


bool CallPython();



};


#endif